from typing import Dict, List, Tuple, Any, Optional, Union

from model.llm import LLMModel
from sql_gen.sqlgen import SQLGenerator
from prompts.prompts_pipeline import get_prompt_version

from utils.utils import get_db_schema_prompt, extract_sql

class CoTSQLGenerator(SQLGenerator):
    """
    A class to generate SQL queries using chain-of-thought (CoT) reasoning.
    This class is designed to take a natural language prompt and convert it into an SQL query,
    using in-context learning inference.
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str = "cot_v0",
        # tables_list: List[str],
    ):
        """
        Initializes the CoTSQLGenerator.

        Args:
            model (LLMModel): The language model to use for SQL generation.
            prompt_version (str): The version of the prompt to use for SQL generation.
        """
        self.model = model
        # self.tables_list = tables_list
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)

    def get_cot_gen_prompt_content(self, 
                                   tables_list: List[str],
                                   ) -> str:
        """
        Formats the main part of the prompt (system message, context, schema, task-specific instructions).
        This content is intended to be used as the 'main_prompt' for the LLM.

        Returns:
            str: The formatted main prompt string for SQL generation.
        """
        db_description = get_db_schema_prompt(self.db_schema, tables_list)
        
        # Construct the prompt using the format with all required placeholders
        prompt = self.sql_gen_prompt_format.format(
            general_task=self.sql_gen_task,
            general_context=self.sql_gen_context,
            db_schema=db_description,
            final_instructions=self.sql_gen_final_instructions,
            cot=self.sql_gen_cot_instructions,
        )

        return prompt

    def generate_sql(self, query: str, tables_list: List[str],
                     ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None,
                     n: int = 1, format_sql: bool = True) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        Generates an SQL query based on the user's natural language query.

        Args:
            query (str): The natural language query from the user. This will be
                        passed as the 'user_prompt' to the LLM.
            tables_list (List[str]): List of tables relevant for the SQL generation.
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            n (int): The number of completions to generate.
            format_sql (bool): Whether to format the SQL query or not. Defaults to True.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing:
                - str: The generated SQL query (extracted from the model's response).
                - dict: The full response dictionary from the language model.
        """
        # Get the main prompt content (system message, schema, instructions)
        cot_gen_prompt = self.get_cot_gen_prompt_content(tables_list=tables_list)

        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ext_knowledge if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+dom_knowledge if dom_knowledge else ""
        # Prepare input for the model
        model_input = {
            "main_prompt": cot_gen_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,  # No few-shot examples for direct generator
        }

        # Generate the SQL query using the model
        model_response = self.model.generate(model_input, n=n)
        
        responses = model_response.get('responses', '')
        cot = {}
        generated_sql = {}
        for k, cot_sql in responses.items():
            cot[k] = cot_sql
            # Format the SQL if requested
            generated_sql[k] = extract_sql(cot_sql, format_sql=format_sql)

        return generated_sql,  { "sql": generated_sql,
                                "cot": cot,
                                "sql_response": model_response,
                                }

    def return_batch(self, query: str, tables_list: List[str],
                     ext_knowledge: Union[str, None], dom_knowledge: Union[str, None],) -> List[Dict[str, str]]:
        """
        Returns the model input for batching requests.

        Args:
            query (str): The natural language query from the user. This will be
                        passed as the 'user_prompt' to the LLM.
            tables_list (List[str]): List of tables relevant for the SQL generation.
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
        Returns:
            dict: The model input dictionary containing the main prompt and user prompt.
        """
              
        # Get the main prompt content (system message, schema, instructions)
        cot_gen_prompt = self.get_cot_gen_prompt_content(tables_list=tables_list)
        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ext_knowledge if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+dom_knowledge if dom_knowledge else ""

        # Prepare input for the model
        model_input = {
            "main_prompt": cot_gen_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,  # No few-shot examples for direct generator
        }

        batch = self.model.return_batch(model_input)

        return batch

    def get_prompt(self, tables_list: List[str] = [""]) -> str:
        """
        Get the SQL generation prompt.

        Args:
            tables_list (List[str]): List of tables relevant for the SQL generation.
        
        Returns:
            str: The formatted SQL generation prompt.
        """
        return self.get_cot_gen_prompt_content(tables_list=tables_list)

    def get_prompts(self, tables_list: List[str] = [""]) -> Dict[str, Any]:
        """
        Get the prompts used in the SQL generation process.

        Returns:
            dict: A dictionary containing the SQL generation prompt and task.
        """
        return { 
            "format": self.sql_gen_prompt_format,
            "db_description": get_db_schema_prompt(self.db_schema, tables_list),
            "task": self.sql_gen_task,
            "context": self.sql_gen_context,
            "final_instructions": self.sql_gen_final_instructions,
            "cot_instructions": self.sql_gen_cot_instructions,
            "prompt": self.get_cot_gen_prompt_content(tables_list=tables_list)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DirectSQLGenerator object to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the DirectSQLGenerator object.
        """
        return {
            "model": self.model.to_dict(),
            "db_schema": self.db_schema,
            "sql_gen_prompt_format": self.sql_gen_prompt_format,
            "sql_gen_task": self.sql_gen_task,
            "sql_gen_context": self.sql_gen_context,
            "sql_gen_final_instructions": self.sql_gen_final_instructions,
            "sql_gen_cot_instructions": self.sql_gen_cot_instructions,
        }
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the DirectSQLGenerator object from a dictionary.
        Args:
            data (Dict[str, Any]): The dictionary representation of the DirectSQLGenerator object.
        """
        self.model = LLMModel.from_dict(data.get("model")) if data.get("model") else None
        self.db_schema = data.get("db_schema")
        self.sql_gen_prompt_format = data.get("sql_gen_prompt_format")
        self.sql_gen_task = data.get("sql_gen_task")
        self.sql_gen_context = data.get("sql_gen_context")
        self.sql_gen_final_instructions = data.get("sql_gen_final_instructions")
        self.sql_gen_cot_instructions = data.get("sql_gen_cot_instructions")
        # Ensure the model is initialized
        if not isinstance(self.model, LLMModel):
            raise ValueError("Model must be an instance of LLMModel")
        # Ensure the db_schema is initialized
        if not isinstance(self.db_schema, dict):
            raise ValueError("Database schema must be a dictionary")
        # Ensure the sql_gen_prompt_format is a string
        if not isinstance(self.sql_gen_prompt_format, str):
            raise ValueError("SQL generation prompt format must be a string")
        # Ensure the sql_gen_task is a string
        if not isinstance(self.sql_gen_task, str):
            raise ValueError("SQL generation task must be a string")
        # Ensure the sql_gen_context is a string
        if not isinstance(self.sql_gen_context, str):
            raise ValueError("SQL generation context must be a string")
        # Ensure the sql_gen_final_instructions is a string
        if not isinstance(self.sql_gen_final_instructions, str):
            raise ValueError("SQL generation final instructions must be a string")
        # Ensure the sql_gen_cot_instructions is a string
        if not isinstance(self.sql_gen_cot_instructions, str):
            raise ValueError("SQL generation CoT instructions must be a string")