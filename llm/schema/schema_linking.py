from typing import Dict, List, Tuple, Any, Union

from model.llm import LLMModel
from prompts.SchemaLinkingPrompts import prompt_schema_linking_v0, sl_final_instructions_v1
from prompts.DBSchemaPrompts import alerce_tables_desc, schema_all_cntxV2_indx
from prompts.prompts_pipeline import get_prompt_version

class SchemaLinking:
    """
    Class to handle the schema linking for a given query.
    
    This class identifies which database tables are relevant to a given
    natural language query.
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str = "sl_v0",
        # db_schema: Union[Dict[str, str], str] = alerce_tables_desc,
        # schema_linking_format: str = prompt_schema_linking_v0,
        # sl_final_instructions: str = sl_final_instructions_v1,
    ):
        """
        Initialize the SchemaLinking class.

        Args:
            model (LLMModel): The language model to use for prediction.
            db_schema (Union[Dict[str, str], str]): The database schema to use for prediction.
            schema_linking_format (str): The format of the schema linking prompt.
            sl_final_instructions (str): The final instructions for the schema linking prompt.
        """
        self.model = model
        self.prompt_version = prompt_version
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)
        # self.db_schema = db_schema
        # self.schema_linking_format = schema_linking_format
        # self.sl_final_instructions = sl_final_instructions
        self.tables_list = None

    def get_schema_descr(self, db_schema: Union[Dict[str, str], str]) -> str:
        """
        Get the tables description from the database schema.

        Args:
            db_schema (Union[Dict[str, str], str]): The database schema to use for the prompt.

        Returns:
            str: The text description of the database schema for linking.
            
        Raises:
            ValueError: If db_schema is not a dict or a str.
        """
        db_description = ""
        if isinstance(db_schema, dict):
            for table_name, table_info in db_schema.items():
                db_description += table_info + "\n"
        elif isinstance(db_schema, str):
            db_description = db_schema
        else:
            raise ValueError("db_schema must be a dict or a str")

        return db_description

    def get_schema_linking_prompt(self) -> str:
        """
        Get the schema linking prompt.

        Returns:
            str: The text prompt for the database schema linking.
        """
        db_description = self.get_schema_descr(self.db_schema)

        prompt = self.schema_linking_format.format(
            db_schema=db_description,
            final_instructions=self.sl_final_instructions,
        )

        return prompt
    
    def tables_list_to_dict(self, tables_str: str) -> List[str]:
        """
        Convert the selected tables string from the model output to a list.

        Args:
            tables_str (str): The selected tables in string format from the model output.
            
        Returns:
            List[str]: The selected tables as a list of table names.
            
        Raises:
            ValueError: If the tables list string is invalid or empty.
        """
        if not tables_str or not isinstance(tables_str, str):
            raise ValueError("Tables list string must be non-empty")
            
        # Handle edge case where response doesn't contain a list format
        if "[" not in tables_str or "]" not in tables_str:
            # Try to find table names in the text
            import re
            potential_tables = re.findall(r'\b([a-zA-Z0-9_]+)\b', tables_str)
            # Filter potential tables to match actual tables in schema
            filtered_tables = [t for t in potential_tables if t in self.db_schema]
            if filtered_tables:
                return filtered_tables
            raise ValueError(f"Could not extract table names from response: {tables_str}")
        
        # Extract content between square brackets
        content = tables_str[tables_str.find("[")+1:tables_str.find("]")]
        if not content.strip():
            raise ValueError("Empty tables list")
            
        # Process the list string into a Python list
        tables_list = content.replace(" ", "").replace("\n", "").strip().split(',')
        tables_list = [table.strip("'\"") for table in tables_list]
        
        # Validate that extracted tables exist in the schema
        print("Extracted tables:", tables_list)
        valid_tables = [table for table in tables_list if table in self.db_schema]
        print("Valid tables:", valid_tables)
        if not valid_tables:
            raise ValueError("No valid tables found in the response")
            
        return valid_tables

    def predict_tables(self, query: str, n: int = 1) -> Tuple[Union[List[List[str]],List[str]], Dict[str, Any]]:
        """
        Predict the table schema for a given query.
        
        Args:
            query (str): The query to predict the table schema linking for.
            n (int): The number of completions to generate.
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: A tuple containing:
                - List[str]: The list of predicted tables
                - Dict[str, Any]: The model's full response
                
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        # Get the schema linking prompt
        schema_linking_prompt = self.get_schema_linking_prompt()

        # Format the input for the model
        model_input = {
            "main_prompt": schema_linking_prompt,
            "user_prompt": query,
            "few_shot_examples": None,
        }
        
        # Predict the tables associated with the query
        pred_tables = self.model.generate(model_input, n=n)
        print(pred_tables)
        try:
            tables_list = []
            # Parse the tables list from the response
            for response in pred_tables['responses']:
                tables_list.append(self.tables_list_to_dict(pred_tables['responses'][response]))
            # Convert the tables list string to a list of table names
            if len(tables_list) == 1:
                self.tables_list = tables_list[0]
            else:
                self.tables_list = tables_list
        except ValueError as e:
            # Add more context to the error
            print(self.tables_list)
            raise ValueError(f"Failed to process schema linking result: {e}")
        # print(type(self.tables_list))
        
        return self.tables_list, pred_tables
    
    def get_prompt_version(self) -> str:
        """
        Get the prompt version used for schema linking.

        Returns:
            str: The prompt version.
        """
        return self.prompt_version

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SchemaLinking object to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the SchemaLinking object.
        """
        return {
            "model": self.model.to_dict(),
            "prompt_version": self.prompt_version,
            "db_schema": self.db_schema,
            "schema_linking_format": self.schema_linking_format,
            "sl_final_instructions": self.sl_final_instructions,
            "tables_list": self.tables_list,
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the SchemaLinking object from a dictionary.
        Args:
            data (Dict[str, Any]): The dictionary representation of the SchemaLinking object.
        """
        self.model = LLMModel.from_dict(data.get("model")) if data.get("model") else None
        self.prompt_version = data.get("prompt_version", "sl_v0")
        self.db_schema = data.get("db_schema")
        self.schema_linking_format = data.get("schema_linking_format")
        self.sl_final_instructions = data.get("sl_final_instructions")
        self.tables_list = data.get("tables_list")
        
        # Ensure the model is initialized
        if not isinstance(self.model, LLMModel):
            raise ValueError("Model must be an instance of LLMModel")
        # Ensure the db_schema is in the correct format
        if not isinstance(self.db_schema, (dict, str)):
            raise ValueError("db_schema must be a dict or a str")
        # Ensure the schema_linking_format is a string
        if not isinstance(self.schema_linking_format, str):
            raise ValueError("schema_linking_format must be a string")
        # Ensure the sl_final_instructions is a string
        if not isinstance(self.sl_final_instructions, str):
            raise ValueError("sl_final_instructions must be a string")
        # Ensure the tables_list is a list
        if not isinstance(self.tables_list, (list, type(None))):
            raise ValueError("tables_list must be a list or None")
        # Ensure the tables_list contains only strings
        if not all(isinstance(table, str) for table in self.tables_list):
            raise ValueError("tables_list must contain only strings")
        