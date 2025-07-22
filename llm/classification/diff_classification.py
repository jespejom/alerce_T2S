from typing import Dict, List, Tuple, Any, Union

from model.llm import LLMModel
from prompts.DiffClassificationPrompts import diff_class_prompt_v1, final_instructions_diff_v1
from prompts.DBSchemaPrompts import schema_all_cntxV2
from prompts.prompts_pipeline import get_prompt_version

from utils.utils import get_db_schema_prompt
from constants import DifficultyLevel, ResponsePrefix

class DifficultyClassification:
    """
    Class to classify the difficulty of a given text query.
    
    This class analyzes natural language queries and determines whether they are
    simple, medium, or advanced in complexity.
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str = "diff_v0",
        # tables_list: List[str],
        # db_schema: Dict[str, str] = schema_all_cntxV2,
        # diff_class_format: str = diff_class_prompt_v1,
        # diff_final_instructions: str = final_instructions_diff_v1,
    ):  
        """
        Initialize the DifficultyClassification class.
        
        Args:
            model (LLMModel): The language model to use for prediction.
            tables_list (List[str]): List of tables relevant for the classification.
            db_schema (Dict[str, str]): The database schema to use for prediction.
            diff_class_format (str): The format of the difficulty classification prompt.
            diff_final_instructions (str): The final instructions for the difficulty classification prompt.
        """
        
        self.model = model
        self.prompt_version = prompt_version
        # self.tables_list = tables_list
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)
        # self.db_schema = db_schema
        # self.diff_class_format = diff_class_format
        # self.diff_final_instructions = diff_final_instructions


    def get_diff_class_prompt(self, 
                              tables_list: List[str],
                              ) -> str:
        """
        Get the difficulty classification prompt.
        The prompt is formatted with the database schema and final instructions.
        
        Returns:
            str: The difficulty classification prompt.
        """
        # get the tables schema
        tables_schema = get_db_schema_prompt(self.db_schema, tables_list)

        prompt = self.diff_class_format.format(
            db_schema=tables_schema,
            final_instructions_diff=self.diff_final_instructions,
        )

        return prompt

    def get_class(self, classification_result: str) -> str:
        """
        Get the class from the classification result.
        The class is determined by the first word of the classification result.
        
        Args:
            classification_result (str): The classification result to parse.

        Returns:
            str: The class of the classification result.
            
        Raises:
            ValueError: If the classification result cannot be mapped to a valid difficulty level.
        """
        # get the class from the classification result
        normalized_result = classification_result.strip().lower().replace(ResponsePrefix.CLASS, "")
        
        valid_classes = DifficultyLevel.get_valid_levels()
        
        if normalized_result in valid_classes:
            return normalized_result
            
        # Fall back to keyword matching if exact match fails
        for difficulty_class in valid_classes:
            # Check if this class appears in the result and no other classes do
            if difficulty_class in normalized_result and all(
                other_class not in normalized_result 
                for other_class in valid_classes if other_class != difficulty_class
            ):
                return difficulty_class
        
        # If no valid class is found, return simple as default and raise a warning
        print(f"Warning: The classification result '{classification_result}' does not match any valid classes. Defaulting to 'simple'.")
        return DifficultyLevel.SIMPLE
        # If the classification result is not valid, raise an error
        # raise ValueError(
        #     f"The classification result {classification_result} is not valid. The result should be '{ResponsePrefix.CLASS} {DifficultyLevel.SIMPLE}', "
        #     f"'{ResponsePrefix.CLASS} {DifficultyLevel.MEDIUM}' or '{ResponsePrefix.CLASS} {DifficultyLevel.ADVANCED}'."
        # )

    def classify_difficulty(self, query: str, tables_list: List[str], n: int = 1) -> Tuple[Union[List[str],str], Dict[str, Any]]:
        """
        Classify the difficulty of the given query.
        The difficulty classification is based on the provided database schema and classes described in the prompt.
        The classes are:
        - Simple: The query is simple and straightforward.
        - Medium: The query is moderately complex and may require some thought with specific knowledge of the database schema.
        - Hard: The query is complex and requires deep understanding of the database schema and SQL language.

        Args:
            query (str): The query to classify.
            tables_list (List[str]): The list of tables relevant for the classification.
            n (int): The number of predictions to generate. Default is 1.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the predicted class and the model's response.
            
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
            
        # get the classification prompt
        classification_prompt = self.get_diff_class_prompt(tables_list)

        # format the prompt
        mssgs_input = {
            "main_prompt": classification_prompt,
            "user_prompt": query,
            "few_shot_examples": None,
        }

        # classify the difficulty of the query
        pred_class = self.model.generate(mssgs_input, n=n)

        # get the classification result
        try:
            diff_list = []
            # Parse the tables list from the response
            for response in pred_class['responses']:
                diff_list.append(self.get_class(pred_class['responses'][response]))
            
            if len(diff_list) == 1:
                classification_result = diff_list[0]
            else:
                classification_result = diff_list
        except ValueError as e:
            # Add more context to the error
            raise ValueError(f"Failed to process classification result: {e}")
        return classification_result, pred_class
    
    def get_prompt(self, tables_list: List[str] = [""]) -> str:
        """
        Get the difficulty classification prompt.
        
        Args:
            tables_list (List[str]): List of tables relevant for the classification.
        
        Returns:
            str: The difficulty classification prompt.
        """
        return self.get_diff_class_prompt(tables_list)
    
    def get_prompts(self, tables_list: List[str] = [""]) -> Dict[str, Any]:
        """
        Get the difficulty classification prompt and final instructions.
        
        Returns:
            dict: A dictionary containing the difficulty classification prompt and final instructions.
        """
        return {
            "format": self.diff_class_format,
            "db_description": get_db_schema_prompt(self.db_schema, tables_list),
            "final_instructions": self.diff_final_instructions,
            "prompt": self.get_diff_class_prompt(tables_list)
        }
    
    def get_prompt_version(self) -> str:
        """
        Get the prompt version used for difficulty classification.
        
        Returns:
            str: The prompt version.
        """
        return self.prompt_version
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DifficultyClassification object to a dictionary.
        
        Returns:
            Dict[str, Any]: The dictionary representation of the DifficultyClassification object.
        """
        return {
            "model": self.model.to_dict(),
            "prompt_version": self.prompt_version,
            "db_schema": self.db_schema,
            "diff_class_format": self.diff_class_format,
            "diff_final_instructions": self.diff_final_instructions,
        }
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the DifficultyClassification object from a dictionary.
        Args:
            data (Dict[str, Any]): The dictionary representation of the DifficultyClassification object.
        """
        self.model = LLMModel.from_dict(data.get("model")) if data.get("model") else None
        self.prompt_version = data.get("prompt_version", "diff_v0")
        self.db_schema = data.get("db_schema")
        self.diff_class_format = data.get("diff_class_format")
        self.diff_final_instructions = data.get("diff_final_instructions")
        
        # Ensure that the model is initialized
        if not isinstance(self.model, LLMModel):
            raise ValueError("Model must be an instance of LLMModel")
        # Ensure the db_schema is in the correct format
        if not isinstance(self.db_schema, (dict, str)):
            raise ValueError("db_schema must be a dict or a str")
        # Ensure the diff_class_format is a string
        if not isinstance(self.diff_class_format, str):
            raise ValueError("diff_class_format must be a string")
        # Ensure the diff_final_instructions is a string
        if not isinstance(self.diff_final_instructions, str):
            raise ValueError("diff_final_instructions must be a string")