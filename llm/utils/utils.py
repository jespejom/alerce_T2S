import re
import os
from constants import SQLErrorType
import pandas as pd
import sqlparse


def load_dataset(
        dataset_path: str,
        dataset_format: str = 'csv',
        ) -> pd.DataFrame:
    """
    Load the dataset from the specified path and format.
    
    Returns:
        pd.DataFrame: The loaded dataset with the expected columns.
    """

    # Determine the dataset path based on the partition
    if dataset_format == "csv":
        data = pd.read_csv(dataset_path)
    elif dataset_format == "json":
        data = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported data format: {dataset_format}")
    
    # Check the dataset has the expected columns
    expected_columns = ['req_id', 'request', 'gold_query', 'external_knowledge', 'domain_knowledge', 
                        'difficulty', 'table_info', 'type']

    for column in expected_columns:
        if column not in data.columns:
            raise ValueError(f"Missing expected column: {column}")
    # return the dataset with the expected columns
    return data[expected_columns]

def get_db_schema_prompt(db_schema: dict, query_tables: list) -> str:
    """
    Get the structure for the specified tables from the database schema.

    Args:
        db_schema (dict): The database schema to use for the prompt.
        query_tables (list): The list of tables used in the query.

    Returns:
        str: The text prompt for the database schema linking.
    """
    schema_description = ""
    # print(query_tables)
    # xd
    for table_name in query_tables:
        if table_name in db_schema.keys():
            schema_description += db_schema[table_name] + "\n"
        else:
            schema_description += "\n"
            print(f"Warning: Table {table_name} not found in the database schema.")

    if schema_description == "":
        raise ValueError(f"No tables {query_tables} found in the database schema")

    return schema_description


def extract_sql(sql_query_response: str, format_sql: bool = True) -> str:
    """
    Extract the SQL query from the model response.

    Args:
        sql_query_response (str): The SQL query to extract.
        format_sql (bool): Whether to format the SQL query or not.
    Returns:
        str: The extracted SQL query.
    """
    # Check if the response contains a code block
    code_block_match = re.search(r"```(sql|SQL)\s*([\s\S]*?)```", sql_query_response)

    if code_block_match:
        sql_code = code_block_match.group(2).strip()  # Get the SQL code
        if format_sql:
            sql_code = sqlparse.format(sql_code, reindent=True, keyword_case='upper')
        return sql_code
    else:
        # If no code block is found, return the response as is
        if format_sql:
            sql_code = sqlparse.format(sql_query_response, reindent=True, keyword_case='upper')
        else:
            sql_code = sql_query_response.strip()
        return sql_code


def get_error_class(error_message: str) -> dict:
    """
    Extract the error class and error message from an ALeRCE's database execution error.
    This function is designed to handle errors from the psycopg2 library, which is commonly used for PostgreSQL database interactions.
    
    Args:
        error_message (str): The error string from the database.
        
    Returns:
        dict: A dictionary containing the error type, error message, and error class.
    """
    error_class = None
    error_line = None
    
    # Extract the error message
    if "psycopg2.errors" in error_message:
        # Extract the error class between "psycopg2.errors." and ")"
        error_match = re.search(r"psycopg2\.errors\.([^)]*)", error_message)
        if error_match:
            error_class = error_match.group(1)
        
        # Extract the line where the error occurred
        error_line_match = re.search(r"\)\s*([^\\]*?)(?:\n\[SQL:|$)", error_message)
        if error_line_match:
            error_line = error_line_match.group(1).strip()
    
    # Determine error type based on content
    if 'timeout' in error_message.lower():
        error_type = SQLErrorType.TIMEOUT
    elif 'not exist' in error_message.lower() or 'does not exist' in error_message.lower():
        error_type = SQLErrorType.UNDEFINED
    else:
        error_type = SQLErrorType.OTHER
    
    return {
        'error_type': error_type,
        'error_message': error_message,
        'error_class': error_class,
        'error_line': error_line
    }
