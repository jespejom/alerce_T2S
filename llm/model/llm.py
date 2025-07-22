from typing import List, Dict, Optional, Union, Any

class LLMModel:
    """
    A class to represent a language model.
    
    This is a base class that should be extended by specific model implementations.
    
    Attributes:
        model_name (str): Name of the language model
        temperature (float): Controls randomness in generation (0.0 = deterministic, higher = more random)
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Nucleus sampling parameter (1.0 = no nucleus sampling)
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 4000, top_p: Optional[float] = None):
        """
        Initialize the language model with specified parameters.
        
        Args:
            model_name (str): Name of the model to use
            temperature (float): Controls randomness in generation (0.0 to 2.0)
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
        """
        self.model_name = model_name
        self.temperature = self._validate_temperature(temperature)
        self.max_tokens = max_tokens if max_tokens > 0 else 4000
        self.top_p = self._validate_top_p(top_p) if top_p is not None else 1.0

    def _validate_temperature(self, temp: float) -> float:
        """
        Validate temperature parameter is within reasonable bounds.
        
        Args:
            temp (float): Temperature value to validate
            
        Returns:
            float: Validated temperature value
        """
        # Most LLMs use temperature between 0.0 and 2.0
        if temp < 0.0:
            return 0.0
        elif temp > 2.0:
            return 2.0
        return temp
        
    def _validate_top_p(self, top_p: float) -> float:
        """
        Validate top_p parameter is within valid range.
        
        Args:
            top_p (float): Top_p value to validate
            
        Returns:
            float: Validated top_p value
        """
        # top_p should be between 0.0 and 1.0
        if top_p < 0.0:
            return 0.0
        elif top_p > 1.0:
            return 1.0
        return top_p

    def generate(self, input_prompt: Dict[str, Any], n: int = 1) -> Dict[str, Any]:
        """
        Generate a text completion using the model.

        Args:
            input_prompt (str or dict): The input prompt to generate text from.
                Can be a string or a dictionary with more complex prompt structure.
            n (int): The number of completions to generate.

        Returns:
            dict: The generated text and other metadata.
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def return_batch(self, input_prompt : Dict[str, Any]) -> Any:
        """
        Returns the model input for batching requests.

        Args:
            input_prompt (str or dict): The input prompt to generate text from.
                Can be a string or a dictionary with more complex prompt structure.

        Returns:
            dict: The model input for batching requests.
            
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def run_batch(self, batch_messages: List[Dict[str, str]], batch_path: str, experiment_name: str, ):
        """
        Run the batch input through the model.

        Args:
            batch_messages (list): List of dictionaries containing batch input messages.
                                    Contains 'batch_id', 'messages', and 'n' (number of completions).
            batch_path (str): The path to the batch input file
            experiment_name (str): The name of the experiment for tracking purposes

        Returns:
            dict: The model output after processing the batch input.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    def get_batch_status(self) -> str:
        """
        Get the status of the batch process.

        Returns:
            str: The status of the batch process.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_batch(self, 
                  batch_response_path: str,
                  ) -> Union[None, Dict[str, Any]]:
        """
        Get the batch prompts used in the SQL generation process.

        Returns:
            dict: A dictionary containing the SQL generation prompt and task.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")        

    def get_batch_response_format(self, chat_completion: Any) -> Dict[str, Any]:
        """
        Extract relevant information from the OpenAI API response.
        
        Args:
            chat_completion: The response from OpenAI API
            
        Returns:
            dict: Formatted response with extracted information
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The name of the model.
        """
        return self.model_name
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters for the model.

        Returns:
            dict: The hyperparameters for the model including name, temperature, max tokens, and top_p.
        """
        return {
            'model_name': self.model_name, 
            'temperature': self.temperature, 
            'max_tokens': self.max_tokens,
            'top_p': self.top_p
        }
    
    def set_hyperparameters(self, 
                          model_name: Optional[str] = None, 
                          temperature: Optional[float] = None, 
                          max_tokens: Optional[int] = None,
                          top_p: Optional[float] = None) -> None:
        """
        Set the hyperparameters for the model.

        Args:
            model_name (str, optional): The name of the model.
            temperature (float, optional): Temperature for the model (0.0 to 2.0).
            max_tokens (int, optional): Maximum number of tokens to generate.
            top_p (float, optional): Nucleus sampling parameter (0.0 to 1.0).
        """
        if model_name is not None:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = self._validate_temperature(temperature)
        if max_tokens is not None:
            self.max_tokens = max_tokens if max_tokens > 0 else self.max_tokens
        if top_p is not None:
            self.top_p = self._validate_top_p(top_p)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the LLMModel object to a dictionary.

        Returns:
            dict: The dictionary representation of the LLMModel object.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the LLMModel object from a dictionary.

        Args:
            data (dict): The dictionary representation of the LLMModel object.
        """
        self.model_name = data.get("model_name", self.model_name)
        self.temperature = data.get("temperature", self.temperature)
        self.max_tokens = data.get("max_tokens", self.max_tokens)
        self.top_p = data.get("top_p", self.top_p)
        # Ensure the model_name is a string
        if not isinstance(self.model_name, str):
            raise ValueError("Model name must be a string")
        # Ensure the temperature is a float
        if not isinstance(self.temperature, float):
            raise ValueError("Temperature must be a float")
        # Ensure the max_tokens is an integer
        if not isinstance(self.max_tokens, int):
            raise ValueError("Max tokens must be an integer")
        # Ensure the top_p is a float or None
        if not isinstance(self.top_p, (float, type(None))):
            raise ValueError("Top_p must be a float or None")