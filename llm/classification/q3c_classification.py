from typing import Dict, List, Tuple, Any

from ..model.llm import LLMModel
from ..prompts.Q3cClassificationPrompts import q3c_class_prompt_v1, final_instructions_q3c_v1
from ..prompts.DBSchemaPrompts import schema_all_cntxV1
from ..utils.utils import get_db_schema_prompt
from ..constants import SpatialClass, ResponsePrefix

class Q3cClassification:
    """
    Class to classify if a given text query is spatial (uses q3c indexing) or not.
    
    This class determines if a natural language query requires spatial operations
    with q3c indexing or not.
    """

    def __init__(
        self,
        model: LLMModel,
        tables_list: List[str],
        db_schema: Dict[str, str] = schema_all_cntxV1,
        q3c_class_format: str = q3c_class_prompt_v1,
        q3c_final_instructions: str = final_instructions_q3c_v1,
    ):
        """
        Initialize the Q3cClassification class.

        Args:
            model (LLMModel): The language model to use for prediction.
            tables_list (List[str]): List of tables relevant for the classification.
            db_schema (Dict[str, str]): The database schema to use for prediction.
                               Defaults to schema_all_cntxV1.
            q3c_class_format (str): The format of the q3c classification prompt.
                                    Defaults to q3c_class_prompt_v1.
            q3c_final_instructions (str): The final instructions for the q3c classification prompt.
                                          Defaults to final_instructions_q3c_v1.
        """
        self.model = model
        self.tables_list = tables_list
        self.db_schema = db_schema
        self.q3c_class_format = q3c_class_format
        self.q3c_final_instructions = q3c_final_instructions

    def get_q3c_class_prompt(self) -> str:
        """
        Get the q3c classification prompt.
        The prompt is formatted with the database schema of the specified tables and final instructions.
        
        Returns:
            str: The q3c classification prompt.
        """
        # Get the schema for the specified tables
        tables_schema = get_db_schema_prompt(self.db_schema, self.tables_list)

        prompt = self.q3c_class_format.format(
            db_schema=tables_schema,
            final_instructions_q3c=self.q3c_final_instructions,
        )

        return prompt

    def get_class(self, classification_result: str) -> str:
        """
        Get the class ("spatial" or "not_spatial") from the classification result.
        The class is determined by the content of the classification result.
        
        Args:
            classification_result (str): The classification result string to parse.

        Returns:
            str: The class of the classification result ("spatial" or "not_spatial").
        
        Raises:
            ValueError: If the classification result is not valid.
        """
        # Normalize the input: remove leading/trailing whitespace, convert to lowercase, remove "class: " prefix
        normalized_result = classification_result.strip().lower().replace(ResponsePrefix.CLASS, "")

        valid_classes = SpatialClass.get_valid_classes()
        
        if normalized_result in valid_classes:
            return normalized_result
            
        # Fall back to keyword matching if exact match fails
        for spatial_class in valid_classes:
            # Check if this class appears in the result and other classes don't
            if spatial_class in normalized_result and all(
                other_class not in normalized_result 
                for other_class in valid_classes if other_class != spatial_class
            ):
                return spatial_class
                
        raise ValueError(
            f"The classification result '{classification_result}' is not valid. Expected "
            f"'{ResponsePrefix.CLASS} {SpatialClass.SPATIAL}' or '{ResponsePrefix.CLASS} {SpatialClass.NOT_SPATIAL}'."
        )

    def classify_q3c(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify if the given query is spatial (q3c) or not_spatial.
        The classification is based on the provided database schema (for selected tables) 
        and classes described in the prompt.
        The classes are:
        - spatial: The query involves spatial indexing or q3c operations.
        - not_spatial: The query does not involve spatial indexing.

        Args:
            query (str): The user query to classify.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the predicted class ("spatial" or "not_spatial") 
                   and the model's raw response dictionary.
                   
        Raises:
            ValueError: If the query is empty or not a string.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        # Get the classification prompt
        classification_prompt = self.get_q3c_class_prompt()

        # Format the messages for the model
        mssgs_input = {
            "main_prompt": classification_prompt,
            "user_prompt": query,
            "few_shot_examples": None,  # Assuming no few-shot examples for now
        }

        # Generate prediction from the model
        pred_output = self.model.generate(mssgs_input)

        # Get the classification class from the model's response
        predicted_class = self.get_class(pred_output['response'])

        return predicted_class, pred_output
