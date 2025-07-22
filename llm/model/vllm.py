import numpy as np
import time
import json
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI
import openai
import multiprocessing
import subprocess

from .llm import LLMModel

# https://github.com/vllm-project/vllm
class vLLMModel(LLMModel):
    """
    Class to interact with vLLM library models.
    
    This class extends the LLMModel base class to provide specific
    implementation for vLLM models.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 4000, top_p: Optional[float] = None):
        """
        Initialize the vLLM model with specified parameters.

        Args:
            model_name (str): Name of the model to use. It should be one of the available models in vLLM.
            temperature (float): Controls randomness (0.0 to 2.0)
            max_tokens (int): Maximum tokens to generate
            top_p (float, optional): Nucleus sampling parameter
        """
        # Initialize the parent class
        super().__init__(model_name, temperature, max_tokens, top_p )

    def call_openai_request(self, model_name: str, messages: List[Dict[str, str]], 
                          t: float, max_tokens: int, top_p: Optional[float] = None,
                          n: int = 1) -> Any:
        """
        Call OpenAI API to generate a text completion with exponential backoff retry.
        
        Args:
            model_name (str): The name of the model to use
            messages (list): List of messages to send to the model
            t (float): Temperature for the model
            max_tokens (int): Maximum number of tokens to generate
            top_p (float, optional): Nucleus sampling parameter
            n (int): The number of completions to generate.
            
        Returns:
            object: The response from the OpenAI API
            
        Raises:
            Exception: If the API call fails after all retry attempts
        """
        # Retry loop for OpenAI request with exponential backoff
        max_retries = 8
        for retry_attempt, delay_secs in enumerate((2**x for x in range(0, max_retries))):
            try:
                # Prepare parameters for the API call
                params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": t,
                    "max_tokens": max_tokens,
                    "n": n,
                }
                
                # Add top_p if it's set
                if top_p is not None:
                    params["top_p"] = top_p

                # Set OpenAI's API key and API base to use vLLM's API server.
                openai_api_key = "EMPTY"
                openai_api_base = "http://localhost:8000/v1"

                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                )

                # Call OpenAI API to generate text completion
                model_response = client.chat.completions.create(**params)
                return model_response
                
            # OpenAI API error: Error code: 404 - {'object': 'error', 'message': 'The model `Qwen/Qwen2.5-1.5B-Instruct` does not exist.', 'type': 'NotFoundError', 'param': None, 'code': 404}.
            except openai.APIError as e:
                # Handle API errors (server issues)
                # wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                wait_time = 1
                print(f"OpenAI API error: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)               

            except openai.RateLimitError as e:
                # Handle rate limiting specifically
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"Rate limit reached: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)

            except openai.AuthenticationError as e:
                # Handle authentication errors
                raise Exception(f"Authentication error: {e}. Please check your API key.")
              
            except openai.APIConnectionError as e:
                # Handle connection errors
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"Connection error: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)
 

            except openai.OpenAIError as e:
                # Handle connection errors
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"OpenAI error: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                # Other errors should be raised immediately without retry
                raise Exception(f"Unexpected error during OpenAI request: {e}")
                
        # If we've exhausted all retries
        raise Exception(f"OpenAI request failed after {max_retries} attempts.")

    def message_format(self, main_prompt: str, user_prompt: Optional[str], 
                     few_shot_examples: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Format the messages for OpenAI API.

        Args:
            main_prompt (str): The main prompt to send as system message
            user_prompt (str, optional): The user prompt to send as user message
            few_shot_examples (list, optional): List of few-shot examples

        Returns:
            list: Formatted messages for OpenAI API
        """
        # Initialize the messages list
        if few_shot_examples:
            for example in few_shot_examples:
                main_prompt += f"\n{example}"
        messages = [{"role": "system", "content": main_prompt}]

        if user_prompt is not None:
            messages.append({"role": "user", "content": user_prompt})
            
        return messages
        
    def get_response_format(self, chat_completion: Any) -> Dict[str, Any]:
        """
        Extract relevant information from the OpenAI API response.
        
        Args:
            chat_completion: The response from OpenAI API
            
        Returns:
            dict: Formatted response with extracted information
        """
        # Check if the response multiple choices
        responses = {}
        for response in chat_completion.choices:
            responses[response.index] = response.message.content

        return {
            'responses': responses,
            'in_toks': chat_completion.usage.prompt_tokens, 
            'out_toks': chat_completion.usage.completion_tokens,
            'id': chat_completion.id, 
            'created': chat_completion.created,
            'model': chat_completion.model
        }

    def run_server(self, command):
        try:
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Server process failed with error: {e}")

    def generate(self, input_prompt: Dict[str, Any], n: int = 1) -> Dict[str, Any]:
        """
        Generate a text completion using OpenAI API.

        Args:
            input_prompt (dict): Dictionary containing 'main_prompt', 'user_prompt', 
                                and optionally 'few_shot_examples'
            n (int): The number of completions to generate.

        Returns:
            dict: Response containing the generated text and metadata
            
        Raises:
            ValueError: If required keys are missing from input_prompt
        """
        # Validate input
        required_keys = ['main_prompt', 'user_prompt']
        for key in required_keys:
            if key not in input_prompt:
                raise ValueError(f"Missing required key in input_prompt: {key}")

        # Format the messages for OpenAI API
        messages = self.message_format(
            input_prompt['main_prompt'], 
            input_prompt['user_prompt'], 
            input_prompt.get('few_shot_examples')
        )

        # Start the vLLM server if not already running
        # vllm serve model_name=self.model_name
        # server_process = multiprocessing.Process(target=self.run_server, args=(f'vllm serve {self.model_name}',))
        # server_process.start()
        # time.sleep(90)  # Wait for the server to start

        # Call OpenAI API to generate a text completion
        response = self.call_openai_request(
            self.model_name, 
            messages, 
            self.temperature, 
            self.max_tokens,
            self.top_p,
            n
        )

        # server_process.terminate()
        # server_process.join()
        
        # Format and return the response
        return self.get_response_format(response)
    

    # def return_batch(self, input_prompt : Dict[str, Any]) -> List[Dict[str, str]]:
    #     """
    #     Returns the model input for batching requests.

    #     Args:
    #         input_prompt (dict): The input prompt dictionary containing 'main_prompt', 
    #                              'user_prompt', and optionally 'few_shot_examples'

    #     Returns:
    #         list: A list of dictionaries formatted for batch processing
    #     """
    #     # Validate input
    #     required_keys = ['main_prompt', 'user_prompt']
    #     for key in required_keys:
    #         if key not in input_prompt:
    #             raise ValueError(f"Missing required key in input_prompt: {key}")

    #     # Format the messages for OpenAI API
    #     messages = self.message_format(
    #         input_prompt['main_prompt'], 
    #         input_prompt['user_prompt'], 
    #         input_prompt.get('few_shot_examples')
    #     )
        
    #     return messages
    
    # def run_batch(self, 
    #               batch_messages: List[Dict[str, str]],
    #               batch_path: str,
    #               experiment_name: str,
    #               ):
    #     """
    #     Run the batch input through the model.

    #     Args:
    #         batch_messages (list): List of dictionaries containing batch input messages.
    #                                 Contains 'batch_id', 'messages', and 'n' (number of completions).
    #         batch_path (str): The path to the batch input file
    #         experiment_name (str): The name of the experiment for tracking purposes

    #     Returns:
    #         dict: The model output after processing the batch input.
    #     """

    #     # Overwrite the batch input file if it exists
    #     with open(batch_path, "w") as f:
    #         f.write("")
    #     # prepare batch input
    #     for batch in batch_messages:
    #         # get the request id and experiment number
    #         custom_id = batch['batch_id'] # batch id with the request and experiment number 

    #         body = {
    #             "model": self.model_name,
    #             "messages": batch["messages"], # list of messages in the Openai api format
    #             "max_tokens": self.max_tokens,
    #             "temperature": self.temperature,
    #             "top_p": self.top_p,
    #             "n": batch['n'], # number of completions to generate
    #         }
    #         # write the batch message to a jsonl file.
    #         with open(batch_path, "a") as f:
    #             f.write(json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n")
        

    #     # prepare batch input
    #     client = openai.OpenAI()
    #     # Upload batch input file
    #     batch_input_file = client.files.create(
    #         file=open(batch_path, "rb"), # jsonl file
    #         purpose="batch"
    #     )
    #     # Define the batch input file ID
    #     self.batch_input_file_id = batch_input_file.id
        

    #     batch_request = client.batches.create(
    #         input_file_id=self.batch_input_file_id,
    #         endpoint="/v1/chat/completions",
    #         completion_window="24h",
    #         metadata={
    #             "description": experiment_name,
    #         }
    #     )
    #     # Define the batch ID
    #     self.batch_id = batch_request.id
    #     print(f"Batch input file uploaded and batch created successfully. The batch ID is {self.batch_id}.")
    
    # def get_batch_status(self) -> str:
    #     """
    #     Get the status of the batch process.

    #     Returns:
    #         str: The status of the batch process.
    #     """
        
    #     client = openai.OpenAI()

    #     batch = client.batches.retrieve(self.batch_input_file_id)

    #     if batch.status == 'completed':
    #         print("Batch completed successfully.")
    #     elif batch.status == 'failed':
    #         print("Batch failed.")
    #     elif batch.status == 'in_progress':
    #         print("Batch is in progress.")
    #     elif batch.status == 'finalizing':
    #         print("Batch is finalizing.")
    #     elif batch.status == 'expired':
    #         print("Batch has expired.")
    #     elif batch.status == 'cancelling':
    #         print("Batch is being cancelled.")
    #     elif batch.status == 'cancelled':
    #         print("Batch has been cancelled.")
    #     else:
    #         print("Batch status is unknown.")
    #     # The status of a given Batch object can be any of the following:

    #     # STATUS	DESCRIPTION
    #     # validating	the input file is being validated before the batch can begin
    #     # failed	the input file has failed the validation process
    #     # in_progress	the input file was successfully validated and the batch is currently being run
    #     # finalizing	the batch has completed and the results are being prepared
    #     # completed	the batch has been completed and the results are ready
    #     # expired	the batch was not able to be completed within the 24-hour time window
    #     # cancelling	the batch is being cancelled (may take up to 10 minutes)
    #     # cancelled	the batch was cancelled
        
    #     return batch.status

    # def get_batch(self, 
    #               batch_response_path: str,
    #               ) -> Union[None, Dict[str, Any]]:
    #     """
    #     Get the batch prompts used in the SQL generation process.

    #     Returns:
    #         dict: A dictionary containing the SQL generation prompt and task.
    #     """
        
    #     client = openai.OpenAI()

    #     batch = client.batches.retrieve(self.batch_id)
    #     # The status of a given Batch object can be any of the following:

    #     # STATUS	DESCRIPTION
    #     # validating	the input file is being validated before the batch can begin
    #     # failed	the input file has failed the validation process
    #     # in_progress	the input file was successfully validated and the batch is currently being run
    #     # finalizing	the batch has completed and the results are being prepared
    #     # completed	the batch has been completed and the results are ready
    #     # expired	the batch was not able to be completed within the 24-hour time window
    #     # cancelling	the batch is being cancelled (may take up to 10 minutes)
    #     # cancelled	the batch was cancelled
        
    #     if batch.status == 'completed':
    #         # Batch is completed, retrieve the output file
    #         file_response = client.files.content(batch.output_file_id)
    #         # The output file is a string containing separated JSON objects for each line. Replace the file if it exists
    #         # return file_response.text
    #         with open(batch_response_path, "w") as f:
    #             f.write(file_response.text)
            
    #         # load the batch response from the jsonl file
    #         with open(batch_response_path, "r") as f:
    #             batch_response_lines = [json.loads(line) for line in f.readlines()]
    #         # Extract the relevant information from the batch response
    #         batch_response = {}
    #         for batch_line in batch_response_lines:
    #            batch_response[batch_line['custom_id']] = self.get_batch_response_format(batch_line)

    #         return batch_response
    #     elif batch.status == 'failed' or batch.status == 'cancelled':
    #         raise Exception(f"Batch failed with status: {batch.status}")
    #     else:
    #         print("Batch is not completed yet. Status: ", batch.status)
    #         return None

    # def get_batch_response_format(self, chat_completion: Any) -> Dict[str, Any]:
    #     """
    #     Extract relevant information from the OpenAI API response.
        
    #     Args:
    #         chat_completion: The response from OpenAI API
            
    #     Returns:
    #         dict: Formatted response with extracted information
    #     """
    #     responses = {}
    #     for response in chat_completion['response']['body']['choices']:
    #         responses[response['index']] = response['message']['content']
    #     return {
    #         'responses': responses,
    #         'in_toks': chat_completion['response']['body']['usage']['prompt_tokens'], 
    #         'out_toks': chat_completion['response']['body']['usage']['completion_tokens'],
    #         'id': chat_completion['response']['body']['id'], 
    #         'created': chat_completion['response']['body']['created'],
    #         'model': chat_completion['response']['body']['model']
    #     }