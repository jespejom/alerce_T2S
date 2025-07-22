from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union, Optional
from model.llm import LLMModel
from utils.utils import get_db_schema_prompt, extract_sql
from prompts.prompts_pipeline import get_prompt_version

class SQLGenerator(ABC):
    """
    Abstract base class for SQL generators.
    
    This class defines the interface that all SQL generators must implement.
    It provides common functionality for generating SQL queries from natural language
    and handles database schema management, prompt construction, and response extraction.
    
    Attributes:
        model (LLMModel): The language model to use for SQL generation
        db_schema (Dict): The database schema information
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str,
        **kwargs
    ):
        """
        Initialize the SQLGenerator.

        Args:
            model (LLMModel): The language model to use for SQL generation.
            prompt_version (str): The version of prompts to use.
            **kwargs: Additional keyword arguments for specific implementations.
        """
        self.model = model
        
        # Load prompts from the specified version
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)

    @abstractmethod
    def generate_sql(self, 
                     query: str, 
                     tables_list: List[str],
                     ext_knowledge: Union[str, None] = None, 
                     dom_knowledge: Union[str, None] = None,
                     n: int = 1, 
                     format_sql: bool = True, 
                     **kwargs) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Generate SQL from a natural language query.

        Args:
            query (str): The natural language query to convert to SQL.
            tables_list (List[str]): List of tables relevant for the query.
            ext_knowledge (Union[str, None], optional): External knowledge to guide generation. Defaults to None.
            dom_knowledge (Union[str, None], optional): Domain knowledge to guide generation. Defaults to None.
            n (int, optional): Number of completions to generate. Defaults to 1.
            format_sql (bool, optional): Whether to format the SQL query. Defaults to True.
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            Tuple[Dict[str, str], Dict[str, Any]]: 
                A tuple containing the generated SQL queries and additional information.
        """
        pass

    @abstractmethod
    def get_prompt(self, tables_list: List[str] = None, **kwargs) -> str:
        """
        Get the prompt used for SQL generation.

        Args:
            tables_list (List[str], optional): List of tables relevant for the query. Defaults to None.
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            str: The formatted prompt.
        """
        pass

    @abstractmethod
    def get_prompts(self, tables_list: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get all prompts used in the SQL generation process.

        Args:
            tables_list (List[str], optional): List of tables relevant for the query. Defaults to None.
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            Dict[str, Any]: A dictionary containing all prompts.
        """
        pass

    def return_batch(self, query: str, tables_list: List[str],
                    ext_knowledge: Union[str, None], dom_knowledge: Union[str, None], 
                    **kwargs) -> List[Dict[str, str]]:
        """
        Return a batch input format for the model.

        Args:
            query (str): The natural language query.
            tables_list (List[str]): List of tables relevant for the query.
            ext_knowledge (Union[str, None]): External knowledge.
            dom_knowledge (Union[str, None]): Domain knowledge.
            **kwargs: Additional arguments specific to the implementation.

        Returns:
            List[Dict[str, str]]: Batch input for the model.
        """
        pass

    def format_extra_knowledge(self, ext_knowledge: Union[str, None], dom_knowledge: Union[str, None]) -> str:
        """
        Format external and domain knowledge into a single string.

        Args:
            ext_knowledge (Union[str, None]): External knowledge to include.
            dom_knowledge (Union[str, None]): Domain knowledge to include.

        Returns:
            str: Formatted knowledge string.
        """
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += f"External Knowledge: {ext_knowledge}" if ext_knowledge else ""
        extra_knowledge += f"\nDomain Knowledge: {dom_knowledge}" if dom_knowledge else ""
        return extra_knowledge

    def prepare_model_input(self, main_prompt: str, query: str, 
                           ext_knowledge: Union[str, None], dom_knowledge: Union[str, None]) -> Dict[str, Any]:
        """
        Prepare the input for the model.

        Args:
            main_prompt (str): The main prompt for the model.
            query (str): The user query.
            ext_knowledge (Union[str, None]): External knowledge.
            dom_knowledge (Union[str, None]): Domain knowledge.

        Returns:
            Dict[str, Any]: The formatted model input.
        """
        extra_knowledge = self.format_extra_knowledge(ext_knowledge, dom_knowledge)
        
        return {
            "main_prompt": main_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,
        }

    def extract_responses(self, model_response: Dict[str, Any], format_sql: bool = True) -> Dict[str, str]:
        """
        Extract and format SQL responses from the model output.

        Args:
            model_response (Dict[str, Any]): The model's response.
            format_sql (bool, optional): Whether to format the SQL. Defaults to True.

        Returns:
            Dict[str, str]: Dictionary of formatted SQL responses.
        """
        generated_sql = model_response.get('responses', {})
        for k in generated_sql.keys():
            # Extract SQL from each response
            generated_sql[k] = generated_sql[k]
            # Format the SQL if requested
            generated_sql[k] = extract_sql(generated_sql[k], format_sql=format_sql)
        
        return generated_sql

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the generator to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the generator.
        """
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the generator from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary representation of the generator.
        """
        pass