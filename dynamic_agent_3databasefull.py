__import__('pysqlite3')

import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
 
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from openai import OpenAI
from typing import Optional
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
import asyncio
from sqlalchemy.exc import SQLAlchemyError
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from pydantic import BaseModel, validator, field_validator, ValidationError
import sqlparse
from google.oauth2 import service_account
from cryptography.fernet import Fernet
import json

class SQLQueryValidator(BaseModel):
    sql_query: str

    if hasattr(BaseModel, "field_validator"):  # Check if Pydantic v2 is available
        @field_validator('sql_query')
        def validate_sql_query(cls, value: str) -> str:
            return cls._validate_sql(value)
    else:
        @validator('sql_query')
        def validate_sql_query(cls, value: str) -> str:
            return cls._validate_sql(value)

    @staticmethod
    def _validate_sql(value: str) -> str:
        value = value.strip()

        if not value or not value.upper().startswith('SELECT'):
            raise ValueError("Only SELECT queries are allowed.")

        parsed = sqlparse.parse(value)
        if not parsed:
            raise ValueError("Invalid SQL syntax.")

        return value

class DatabaseAgent(Workflow):
    def __init__(self, vn, db_type: str, db_credentials: dict, timeout: Optional[float] = 200.0):
        super().__init__(timeout=timeout)
        self.vn = vn
        self.db_type = db_type
        self.db_credentials = db_credentials
        self.engine = self.create_db_connection()
        self.vn.run_sql = self.run_query
        self.vn.run_sql_is_set = True
        

    def create_db_connection(self):
        """Create a database connection dynamically based on the selected type."""
        if self.db_type == "AWS Athena":
            aws_access_key = self.db_credentials.get("aws_access_key")
            aws_secret_key = self.db_credentials.get("aws_secret_key")
            region_name = self.db_credentials.get("region_name")
            db_name = self.db_credentials.get("db_name")
            s3_output_location = self.db_credentials.get("s3_output_location")

            return create_engine(
                f'awsathena+rest://{aws_access_key}:{aws_secret_key}@athena.{region_name}.amazonaws.com/{db_name}?s3_staging_dir={s3_output_location}'
            )
        elif self.db_type == "Azure Synapse":
            server = self.db_credentials.get("server")
            database = self.db_credentials.get("database")
            username = self.db_credentials.get("username")
            password = self.db_credentials.get("password")

            connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};' \
                    f'SERVER={server};' \
                    f'DATABASE={database};' \
                    f'UID={username};' \
                    f'PWD={password};'

             # Create the SQLAlchemy engine
            return create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")
        
        elif self.db_type == "GCP BigQuery":
             project_id = self.db_credentials.get("project_id")
             dataset_name = self.db_credentials.get("dataset_name")
             # Path to your service account key
             service_account_key = r'colab-poc-444604-72bf522b92ea.json'
             credentials = service_account.Credentials.from_service_account_file(service_account_key)
             # Create the SQLAlchemy engine URL with the service account credentials
             engine_url = f"bigquery://{project_id}/{dataset_name}?credentials_path={service_account_key}"

             # Create SQLAlchemy engine
             engine = create_engine(engine_url)
             
             return engine

        else:
            raise ValueError("Unsupported database type")

    def run_query(self, sql: str) -> pd.DataFrame:
        """Execute a query against the database and return the result as a DataFrame."""
        try:
            # Validate the SQL query using Pydantic
            SQLQueryValidator(sql_query=sql)

            with self.engine.connect() as connection:
                result = connection.execute(text(sql))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

        except ValidationError as e:
            raise ValueError(f"SQL Query Validation Failed: {e}")

        except SQLAlchemyError as e:
            raise ValueError(f"Database Query Error: {e}")

    

    def engineer_prompt(self, question: str, db_type: str) -> str:
        """Generate a prompt for SQL query generation based on the schema, user question, and database type."""
       
        # Define database-specific instructions
        db_guidelines = {
            "GCP BigQuery": """
    - Choose the correct table for the query based on its schema or context from the user query.
    - Use `LIMIT` to restrict the number of rows in the output.
    - Use `SAFE_CAST` when type conversions are needed.
    - Fully qualify table names in the format `project_id.dataset_id.table_name`.
    - Use standard SQL syntax for BigQuery.
    """,
            "AWS Athena": """
    - Choose the correct table for the query based on its schema or context from the user query.
    - Use `LIMIT` to restrict the number of rows in the output.
    - Use ANSI SQL-compatible syntax supported by AWS Athena.
    - Athena requires table and column names to be case-sensitive.
    - Use the `FROM database_name.table_name` format to qualify table names.
    - For filtering text columns based on user queries:
      - **Prefer `LIKE '%keyword%'`** when searching for words in text columns.
      - If available, **use `CONTAINS(column_name, 'keyword')`** for full-text search.
      - **Do NOT use `=` for text-based searches** unless checking for an exact match.
    - Use `DISTINCT` to ensure unique values where necessary.
    """,
           "Azure Synapse":"""

    - Choose the correct table for the query based on its schema or context from the user query.
    - Use TOP instead of LIMIT to restrict the number of rows in the output. Do not use LIMIT.
    - Use fully qualified table and column names where applicable.
    - Ensure that date functions and string concatenation follow SQL Server syntax.
    - Use OFFSET FETCH for pagination instead of LIMIT."""
        }
       
        # Database-specific instructions
        db_specific_instructions = db_guidelines.get(db_type, "No specific guidelines for the selected database.")
 
        # Combine the base prompt with database-specific instructions
        return f"""
        You are a {db_type} SQL expert tasked with generating a query based on the provided schema. It is critical to adhere strictly to the schema and use the exact table and column names as specified.
 
    **Task**:
    1. **Objective**: Write an SQL query to address the following question:
    *"{question}"*
 
    2. **Pre-query Requirements**:
    - **Read-only operation**: The query must only perform read operations. No modifications such as `INSERT`, `UPDATE`, or `DELETE` are permitted.
    - **Schema Validation**: Carefully review the schema to:
        - Identify the correct tables and columns relevant to the question.
        - Verify all table and column names match exactly as specified in the schema.
        - Understand the relationships between tables (if applicable).
        - **Always use `GROUP BY` for all the queries** (if aggregation is required).
        - Ensure that **all column names are fully qualified** using the format `table_name.column_name` to avoid ambiguity.
 
    3. **Database-Specific Guidelines**:
    {db_specific_instructions}
 
    4. **Query Construction Guidelines**:
    - All the queries are asked in the context of Nexturn,therefore omit the word "Nexturn"
    - Use only the table names, column names, and relationships explicitly provided in the schema.
    - Fully qualify all column names with their respective table names, especially when columns with the same name appear in multiple tables.
    - If the schema does not contain all the necessary information for the query, provide a detailed explanation of why the query cannot be completed.
    - Use `AS` to alias columns or tables when appropriate for clarity, but do not deviate from the schema.
    - Convert the following natural language query into an SQL query. When filtering text columns, use the CONTAINS clause instead of the WHERE clause with = or LIKE, unless an exact match is explicitly requested. Ensure the query is correctly formatted for execution.
    - Use the `FROM database_name.table_name` format to qualify table names.
    - For filtering text columns based on user queries:
      - **Prefer `LIKE '%keyword%'`** when searching for words in text columns.
      - If available, **use `CONTAINS(column_name, 'keyword')`** for full-text search.
      - **Do NOT use `=` for text-based searches** unless checking for an exact match.
    - Use `DISTINCT` to ensure unique values where necessary.
    -Example:list down all the FP contracts at Nexturn-"SELECT * FROM your_tableWHERE your_column LIKE '%FP%"
 
    5. **Validation Checklist**:
    - Double-check that all table and column names in the query exactly match the schema.
    - Ensure that relationships between tables (if used) align with those described in the schema.
    - Confirm that the query fully addresses the question within the constraints of the schema.
    - Confirm that all column references are fully qualified to prevent ambiguity errors.
    - Verify that text searches use CONTAINS() for better performance with long text fields.
 
    6. **Output**:
    - If the query can be written, return only the SQL query.
    - If the query cannot be generated due to insufficient or unclear schema information, provide a detailed rationale for why it is not possible.
 
    **Note**: Adherence to the schema is mandatory. Queries that do not align with the schema, fail to disambiguate column names, or include assumptions will be considered invalid. Return only the SQL query.
        """
    
    @step
    async def start_chat(self, ev: StartEvent) -> StopEvent:
        question = ev.topic
        try:
            #schema = self.get_schema()
            prompt = self.engineer_prompt(question,self.db_type)
            
            sql_query = self.vn.generate_sql(prompt)
            st.write("### Generated SQL Query:")
            st.code(sql_query, language='sql')

            try:
                SQLQueryValidator(sql_query=sql_query)
                st.success("The Query Is Verified")
            except ValueError as e:
                st.error(f"Validation Failed")
                return StopEvent(result="Please Refine Your Question")

            # Check if the query is a Modification query (not SELECT)
            if any(keyword in sql_query.strip().upper() for keyword in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "MERGE", "REPLACE", "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT"]):
                st.warning("Warning: The generated SQL query contains modification operations. It will not be executed.")
                return StopEvent(result="Query not executed due to potential modification operations.")

            # Execute the valid SELECT query
            st.success("The Query is allowed to run")
            st.success("Executing..........")
            result_df = self.run_query(sql_query)

            plotly_code = self.vn.generate_plotly_code(question=question, sql=sql_query, df=result_df)
            fig = self.vn.get_plotly_figure(plotly_code=plotly_code, df=result_df)

            return StopEvent(result=[result_df, fig])

        except Exception as e:
            st.error(f"Error: {e}")
            return StopEvent(result="An error occurred while processing your query. Please try again.")

def main():
    class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
        def __init__(self, config=None):
            ChromaDB_VectorStore.__init__(self, config=config)
            OpenAI_Chat.__init__(self, config=config)

    st.title("Dynamic Query Interface")
    st.write("Select a database, enter credentials, and ask your questions!")

    db_type = st.selectbox("Choose Database Type", ["AWS Athena", "Azure Synapse", "GCP BigQuery"])

    db_credentials = {}
    if db_type == "AWS Athena":
        db_credentials["aws_access_key"] = st.text_input("AWS Access Key")
        db_credentials["aws_secret_key"] = st.text_input("AWS Secret Key", type="password")
        db_credentials["region_name"] = st.text_input("AWS Region")
        db_credentials["db_name"] = st.text_input("Database Name")
        db_credentials["s3_output_location"] = st.text_input("S3 Output Location")
        embedding_path='embedding_aws_matrix2'
    elif db_type == "Azure Synapse":
        db_credentials["server"] = st.text_input("Server")
        db_credentials["database"] = st.text_input("Database Name")
        db_credentials["username"] = st.text_input("Username")
        db_credentials["password"] = st.text_input("Password", type="password")
        embedding_path='embedding_azure'
    elif db_type == "GCP BigQuery":
        db_credentials["project_id"] = st.text_input("Project ID")
        db_credentials["dataset_name"] = st.text_input("Dataset Name")
        embedding_path='embedding_gcp_14table1'

    if st.button("Connect"):
        #st.write("Selected Embedding:", embedding_path)
        with open('secret.key', 'rb') as key_file:
                key = key_file.read()
    
                cipher_suite = Fernet(key)
    
            # Load the encrypted configuration data
        with open('config.enc', 'r') as config_file:
                encrypted_data = json.load(config_file)
        decrypted_credentials = json.loads(cipher_suite.decrypt(encrypted_data).decode())
        API_KEY = decrypted_credentials.get("api_key", "")
    
            # Decrypt the sensitive information
        data = {key: cipher_suite.decrypt(value.encode()).decode() for key, value in encrypted_data.items()}
        vn = MyVanna(config={'api_key': API_KEY, 
                             'model': 'gpt-3.5-turbo', 'temperature': 0.2,'path': embedding_path})
        database_agent = DatabaseAgent(vn=vn, db_type=db_type, db_credentials=db_credentials)
        st.session_state["database_agent"] = database_agent
        
    if "database_agent" in st.session_state:
        database_agent = st.session_state["database_agent"]

        user_input = st.text_input("Enter your question (or leave blank to exit):")

        if user_input:
            async def process_input():
                start_event = StartEvent(topic=user_input)
                stop_event = await database_agent.start_chat(start_event)

                if stop_event != None:
                    if isinstance(stop_event.result, list):
                        result_df, fig = stop_event.result
                        st.write("### Query Result:")
                        st.dataframe(result_df)

                        st.write("### Generated Plot:")
                        st.plotly_chart(fig)
                    else:
                        st.write("### Result:")
                        st.write(stop_event.result)
                else:
                    st.write("No result available. Please try again.")

            asyncio.run(process_input())

if __name__ == "__main__":
    main()
