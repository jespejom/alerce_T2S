from typing import Dict, List, Tuple, Any, Optional

from model.llm import LLMModel
from prompts.SelfCorrectionPrompts import (
    general_task_selfcorr_v1, 
    general_context_selfcorr_v1, 
    final_instr_selfcorr_v0,
    self_correction_timeout_prompt_v2, 
    self_correction_no_exist_prompt_v2, 
    self_correction_schema_prompt_v2
)
from prompts.DBSchemaPrompts import schema_all_cntxV2
from prompts.prompts_pipeline import get_prompt_version

from utils.utils import get_db_schema_prompt, extract_sql, get_error_class
from constants import SQLErrorType

class SelfCorrection:
    """
    Class to correct SQL queries based on execution errors.
    Uses a language model to analyze the error and suggest corrections.
    """

    def __init__(
        self,
        model: LLMModel,
        tables_list: List[str],
        prompt_version: str = "dir_v0",
        # db_schema: Dict[str, str] = schema_all_cntxV2, 
        # general_task: str = general_task_selfcorr_v1,
        # general_context: str = general_context_selfcorr_v1,
        # final_instructions: str = final_instr_selfcorr_v0,
        # timeout_prompt_format: str = self_correction_timeout_prompt_v2,
        # not_exist_prompt_format: str = self_correction_no_exist_prompt_v2,
        # schema_error_prompt_format: str = self_correction_schema_prompt_v2,
    ):
        """
        Initialize the SelfCorrection class.

        Args:
            model (LLMModel): The language model to use for corrections.
            db_schema (Dict[str, str]): The database schema to use for reference.
            tables_list (List[str]): List of tables relevant for the query.
            general_task (str): General description of the self-correction task.
            general_context (str): Context information about the database.
            final_instructions (str): Final instructions for the correction process.
            timeout_prompt_format (str): Prompt format for timeout errors.
            not_exist_prompt_format (str): Prompt format for not_exist errors.
            schema_error_prompt_format (str): Prompt format for schema errors.
        """
        self.model = model
        self.tables_list = tables_list
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)
        # self.db_schema = db_schema
        # self.general_task = general_task
        # self.general_context = general_context
        # self.final_instructions = final_instructions
        # self.timeout_prompt_format = timeout_prompt_format
        # self.not_exist_prompt_format = not_exist_prompt_format
        # self.schema_error_prompt_format = schema_error_prompt_format

    def get_correction_prompt(self, query: str, sql_query: str, sql_error: str) -> str:
        """
        Get the appropriate correction prompt based on error type.

        Args:
            query (str): The original user query.
            sql_query (str): The SQL query that caused the error.
            sql_error (str): The error message from the database.

        Returns:
            str: The formatted prompt for the specified error type.
        """
        # Get the database schema description for the relevant tables
        db_description = get_db_schema_prompt(self.db_schema, self.tables_list)
        
        # Determine error type from the error message
        error_info = get_error_class(sql_error)
        error_type = error_info['error_type']
        # Use the cleaned error message
        clean_error = error_info['error_message']

        # Add context about the error class if available
        error_context = ""
        if error_info.get('error_class'):
            error_context = f"Error class: {error_info['error_class']}\n"
        
        # Add context about the error line if available
        if error_info.get('error_line'):
            error_context += f"Error occurred at: {error_info['error_line']}\n"

        # Select the appropriate prompt format based on error type
        if error_type == SQLErrorType.TIMEOUT:
            prompt_format = self.timeout_prompt_format
        elif error_type == SQLErrorType.NOT_EXIST:
            prompt_format = self.not_exist_prompt_format
        else:  # Default to schema error prompt for other errors
            prompt_format = self.schema_error_prompt_format

        # Format the prompt with the specific details
        prompt = prompt_format.format(
            Self_correction_task=self.general_task,
            request=query,
            tab_schema=db_description,
            sql_query=sql_query,
            sql_error=f"{error_context}{clean_error}",
            final_instructions=self.final_instructions,
            context=self.general_context
        )

        return prompt

    def correct_query(self, query: str, sql_query: str, sql_error: str, n: int =1, format_sql: bool = True) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Generate a corrected SQL query based on the error.

        Args:
            query (str): The original user query.
            sql_query (str): The SQL query that caused the error.
            sql_error (str): The error message from the database.
            n (int): The number of corrections to generate.
            format_sql (bool): Whether to format the SQL query or not.

        Returns:
            Tuple[Dict[str, str], Dict[str, Any]]: A tuple containing:
                - Dict[str, str]: Dictionary of corrected SQL queries with keys as response ids.
                - Dict[str, Any]: The full response from the model.
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("Original query cannot be empty")
        if not sql_query or not sql_query.strip():
            raise ValueError("SQL query cannot be empty")
        if not sql_error or not sql_error.strip():
            raise ValueError("SQL error message cannot be empty")
            
        # Get the appropriate correction prompt
        correction_prompt = self.get_correction_prompt(query, sql_query, sql_error)

        # Format the messages for the model
        model_input = {
            "main_prompt": correction_prompt,
            "user_prompt": None,  # No additional user prompt needed since everything is in the main prompt
            "few_shot_examples": None,
        }

        # Generate the corrected SQL query
        pred_output = self.model.generate(model_input, n=n)
        
        # Extract the corrected SQL query from the response
        corrected_sql = pred_output.get('responses', {})
        for k in corrected_sql.keys():
            # Extract SQL from each response
            corrected_sql[k] = extract_sql(corrected_sql[k], format_sql=format_sql)
        
        # Return the corrected SQL query and the model's response
        return corrected_sql, pred_output
    
    def return_batch(self, query: str, sql_query: str, sql_error: str) -> List[Dict[str, str]]:
        """
        Returns the model input for batching requests.

        Args:
            query (str): The original user query.
            sql_query (str): The SQL query that caused the error.
            sql_error (str): The error message from the database.

        Returns:
            List[Dict[str, str]]: A list with the model input dictionary for batch processing.
        """
        # Get the appropriate correction prompt
        correction_prompt = self.get_correction_prompt(query, sql_query, sql_error)

        # Format the messages for the model
        model_input = {
            "main_prompt": correction_prompt,
            "user_prompt": None,  # No additional user prompt needed since everything is in the main prompt
            "few_shot_examples": None,
        }

        # Get batch format from the model
        batch = self.model.return_batch(model_input)
        
        return batch
    
    def get_prompts(self) -> Dict[str, str]:
        """
        Returns the prompts used by this self-correction instance.

        Returns:
            Dict[str, str]: A dictionary containing all the prompts used by this instance.
        """
        return {
            "general_task": self.general_task,
            "general_context": self.general_context,
            "final_instructions": self.final_instructions,
            "timeout_prompt_format": self.timeout_prompt_format,
            "not_exist_prompt_format": self.not_exist_prompt_format,
            "schema_error_prompt_format": self.schema_error_prompt_format,
            # full prompts
            "full_timeout_prompt": self.timeout_prompt_format.format(
                Self_correction_task=self.general_task,
                request="",
                tab_schema="",
                sql_query="",
                sql_error="",
                final_instructions=self.final_instructions,
                context=self.general_context
            ),
            "full_not_exist_prompt": self.not_exist_prompt_format.format(
                Self_correction_task=self.general_task,
                request="",
                tab_schema="",
                sql_query="",
                sql_error="",
                final_instructions=self.final_instructions,
                context=self.general_context
            ),
            "full_schema_error_prompt": self.schema_error_prompt_format.format(
                Self_correction_task=self.general_task,
                request="",
                tab_schema="",
                sql_query="",
                sql_error="",
                final_instructions=self.final_instructions,
                context=self.general_context
            ),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the self-correction instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of this instance.
        """
        return {
            "model": self.model.to_dict(),
            "db_schema": self.db_schema,
            "tables_list": self.tables_list,
            "general_task": self.general_task,
            "general_context": self.general_context,
            "final_instructions": self.final_instructions,
            "timeout_prompt_format": self.timeout_prompt_format,
            "not_exist_prompt_format": self.not_exist_prompt_format,
            "schema_error_prompt_format": self.schema_error_prompt_format,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Updates this instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the attributes to update.
        """
        self.model = LLMModel.from_dict(data["model"])
        self.db_schema = data["db_schema"]
        self.tables_list = data["tables_list"]
        self.general_task = data["general_task"]
        self.general_context = data["general_context"]
        self.final_instructions = data["final_instructions"]
        self.timeout_prompt_format = data["timeout_prompt_format"]
        self.not_exist_prompt_format = data["not_exist_prompt_format"]
        self.schema_error_prompt_format = data["schema_error_prompt_format"]

        if not isinstance(self.model, LLMModel):
            raise ValueError("Model must be an instance of LLMModel")
        if not isinstance(self.db_schema, dict):
            raise ValueError("Database schema must be a dictionary")
        if not isinstance(self.tables_list, list):
            raise ValueError("Tables list must be a list")
        if not isinstance(self.general_task, str):
            raise ValueError("General task must be a string")
        if not isinstance(self.general_context, str):
            raise ValueError("General context must be a string")
        if not isinstance(self.final_instructions, str):
            raise ValueError("Final instructions must be a string")
        if not isinstance(self.timeout_prompt_format, str):
            raise ValueError("Timeout prompt format must be a string")
        if not isinstance(self.not_exist_prompt_format, str):
            raise ValueError("Not exist prompt format must be a string")
        if not isinstance(self.schema_error_prompt_format, str):
            raise ValueError("Schema error prompt format must be a string")