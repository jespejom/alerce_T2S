from urllib import response
import google
from google import genai
from google.genai import types
import time

import numpy as np
import json
from typing import Dict, List, Optional, Any, Union

from .llm import LLMModel


class GeminiModel(LLMModel):
    """
    Class to interact with Google's Gemini API for text generation.
    
    This class extends the LLMModel base class to provide specific
    implementation for Google's Gemini models.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 4000, top_p: Optional[float] = None):
        """
        Initialize the Gemini model with specified parameters.
        
        Args:
            model_name (str): Name of the Gemini model to use (e.g., 'gemini-2.5-flash', 'gemini-2.5-pro')
            temperature (float): Controls randomness (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate 
            top_p (float, optional): Nucleus sampling parameter
        """
        # Initialize the parent class
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.client = None

    def _get_client(self) -> genai.Client:
        """
        Create or return the Gemini client.
        
        Returns:
            genai.GeminiClient: The Gemini client instance
        """
        if self.client is None:
            self.client = genai.Client()
        return self.client

    def call_gemini_request(self, model_name: str, system_msg: Optional[str], messages: List[Dict[str, str]], 
                               t: float, max_tokens: int, top_p: Optional[float] = None,
                               ) -> Any:
        """
        Call Gemini API to generate a text completion with exponential backoff retry.
        
        Args:
            model_name (str): The name of the model to use
            system_msg (str, optional): System message for the model
            messages (list): List of messages to send to the model
            t (float): Temperature for the model
            max_tokens (int): Maximum number of tokens to generate
            top_p (float, optional): Nucleus sampling parameter
            n (int): The number of completions to generate.
            
        Returns:
            object: The response from the Anthropic API
            
        Raises:
            Exception: If the API call fails after all retry attempts
        """
        max_retries = 6
        for retry_attempt, delay_secs in enumerate((2**x for x in range(0, max_retries))):
            try:
                client = self._get_client()
                
                # Prepare request parameters
                params = {
                    "model": model_name,
                    "contents": messages,
                    # "temperature": t,
                    # "max_tokens": max_tokens,
                    # "messages": messages
                }
                # Add top_p if it's set
                # if top_p is not None:
                #     params["top_p"] = top_p
                
                # Add system message if provided
                if system_msg is not None:
                    params["config"] = types.GenerateContentConfig(
                        system_instruction=system_msg,
                        temperature=t,
                        max_output_tokens=max_tokens,
                        # thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                        )
                else:
                    params["config"] = types.GenerateContentConfig(
                        temperature=t,
                        max_output_tokens=max_tokens,
                        # thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                        )
                
                # Make the API call
                model_response = client.models.generate_content(**params
                                                                )
                
                return model_response

            except google.genai.errors.ServerError as e:
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"Server error: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)    
            
            except google.api_core.exceptions.ServiceUnavailable as e:
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"Connection error: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                print(f"Cause: {e.__cause__}")
                time.sleep(wait_time)
                
            # RESOURCE_EXHAUSTED
            except google.api_core.exceptions.ResourceExhausted as e:
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"Rate limit reached: {e}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)
                
            except google.api_core.exceptions.GoogleAPICallError as e:
                wait_time = delay_secs + (np.random.randint(0, 1000) / 1000.0)
                print(f"API Status error: {e}. Status code: {e.status_code}. Retrying in {round(wait_time, 2)} seconds. Attempt {retry_attempt+1}/{max_retries}")
                time.sleep(wait_time)
                
            except Exception as e:
                # For any other exception, don't retry
                raise Exception(f"Unexpected error during Gemini API call: {e}")
                
        # If all retries fail
        raise Exception(f"Gemini API request failed after {max_retries} attempts.")
    
    def message_format(self, main_prompt: Optional[str], user_prompt: Optional[str], few_shot_examples: Optional[List[str]] = None,) -> tuple:
        """
        Format the messages for Gemini API.

        Args:
            main_prompt (str, optional): The main prompt to send as system message
            user_prompt (str, optional): The user prompt to send as user message
            few_shot_examples (list, optional): List of few-shot examples

        Returns:
            tuple: (system_message, messages) formatted for Gemini API
        """
        messages = []
        system_msg = None
        
        # Combine few-shot examples into a string if provided
        few_shot_string = ""
        if few_shot_examples:
            few_shot_string = "\n".join(few_shot_examples)
        
        # Determine what to use as system message and user content
        if main_prompt is not None:
            # Use main_prompt as system message

            # Add user content if provided
            if user_prompt is not None:
                messages.append(user_prompt)
                if few_shot_string:
                    system_msg = f"{main_prompt}\n{few_shot_string}"
                else:
                    system_msg = main_prompt
            else:
                system_msg = main_prompt
                if few_shot_string:
                    # If few-shot examples are provided, append them to the system message
                    system_msg = f"{system_msg}\n{few_shot_string}"
                # If no user_prompt, use an empty message
                messages.append(system_msg)
                system_msg = None  # Clear system message to avoid duplication
        else:
            # No system message, put everything in user message
            user_content = user_prompt if user_prompt is not None else ""
            if few_shot_string:
                user_content = f"{user_content}\n{few_shot_string}" if user_content else few_shot_string
                
            messages.append(user_content)
            
        return system_msg, messages
    
    def get_response_format(self, chat_completion: List) -> Dict[str, Any]:
        """
        Extract relevant information from the Gemini API response.
        
        Args:
            chat_completion: The response from Gemini API
            
        Returns:
            dict: Formatted response with extracted information
        """
        responses = {}
        responses_text = {}
        ids = {}
        usage = {}
        in_toks = 0
        out_toks = 0
        # print("chat_completion:", chat_completion)
        # print("type chat_completion:", type(chat_completion))
        # print("len chat_completion:", len(chat_completion))
        # print("candidates:", chat_completion[0].candidates)
        # print("model_version:", chat_completion[0].model_version)
        # print("response_id:", chat_completion[0].response_id)
        # print("usage_metadata:", chat_completion[0].usage_metadata)
        # print("text:", chat_completion[0].text)
        # print("text:", chat_completion[0].usage_metadata.prompt_token_count)
        # print("text:", chat_completion[0].usage_metadata.candidates_token_count)

#         chat_completion: [GenerateContentResponse(
#   automatic_function_calling_history=[],
#   candidates=[
#     Candidate(
#       avg_logprobs=-0.08666654229164124,
#       content=Content(
#         parts=[
#           Part(
#             text="""['object', 'probability', 'step']
# """
#           ),
#         ],
#         role='model'
#       ),
#       finish_reason=<FinishReason.STOP: 'STOP'>  
#     ),
#   ],
#   model_version='gemini-2.0-flash-lite',
#   response_id='GhQgaZzZHYO6qtsPtdbA6Aw',
#   sdk_http_response=HttpResponse(
#     headers=<dict len=11>
#   ),
#   usage_metadata=GenerateContentResponseUsageMetadata(
#     candidates_token_count=10,
#     candidates_tokens_details=[
#       ModalityTokenCount(
#         modality=<MediaModality.TEXT: 'TEXT'>,
#         token_count=10
#       ),
#     ],
#     prompt_token_count=623,
#     prompt_tokens_details=[
#       ModalityTokenCount(
#         modality=<MediaModality.TEXT: 'TEXT'>,   
#         token_count=623
#       ),
#     ],
#     total_token_count=633
#   )
# )]
        for i, response in enumerate(chat_completion):
            responses[i] = response.candidates[0].content.parts[0].text
            responses_text[i] = response.text
            ids[i] = response.response_id
            in_toks += response.usage_metadata.prompt_token_count
            out_toks += response.usage_metadata.candidates_token_count
            usage[i] = dict(response.usage_metadata)

        return {
            'responses': responses,
            'responses_text': responses_text,
            'in_toks': in_toks,
            'out_toks': out_toks,
            'id': ids,
            'created': ids, 
            'model': self.model_name,
        }
    
    def generate(self, input_prompt: Dict[str, Any], n: int = 1) -> Dict[str, Any]:
        """
        Generate a text completion using Anthropic API.

        Args:
            input_prompt (dict): Dictionary containing 'main_prompt', 'user_prompt', 
                               and optionally 'few_shot_examples'
            n (int): The number of completions to generate.

        Returns:
            dict: Response containing the generated text and metadata
            
        Raises:
            ValueError: If both main_prompt and user_prompt are missing
        """
        # Check that at least one of main_prompt or user_prompt is provided
        if input_prompt.get('main_prompt') is None and input_prompt.get('user_prompt') is None:
            raise ValueError("At least one of 'main_prompt' or 'user_prompt' must be provided")

        
        responses = []
        for n_response in range(n):
            # Format the messages for Anthropic API
            if n_response==0 and n:
                system_msg, messages = self.message_format(
                    input_prompt.get('main_prompt'), 
                    input_prompt.get('user_prompt'), 
                    input_prompt.get('few_shot_examples')
                )
            else:
                system_msg, messages = self.message_format(
                    input_prompt.get('main_prompt'), 
                    input_prompt.get('user_prompt'), 
                    input_prompt.get('few_shot_examples'),
                )
            # Call the model with the prompt and db_schema
            model_response = self.call_gemini_request(
                self.model_name,
                system_msg, 
                messages,
                self.temperature,
                self.max_tokens,
                self.top_p,
            )
            responses.append(model_response)

        return self.get_response_format(responses)

    def return_batch(self, input_prompt : Dict[str, Any]) -> tuple:
        """
        Returns the model input for batching requests.

        Args:
            input_prompt (dict): The input prompt dictionary containing 'main_prompt', 
                                 'user_prompt', and optionally 'few_shot_examples'

        Returns:
            tuple: A tuple containing:
                - system_msg (str): The system message for the model
                - messages (list): A list of dictionaries formatted for batch processing
        """
        raise NotImplementedError("Batch processing is not implemented for Gemini API.")
        # TODO: Implement batch processing for Gemini API

        # Check that at least one of main_prompt or user_prompt is provided
        if input_prompt.get('main_prompt') is None and input_prompt.get('user_prompt') is None:
            raise ValueError("At least one of 'main_prompt' or 'user_prompt' must be provided")

        # Get the main prompt content (system message, schema, instructions)
        system_msg, messages = self.message_format(
            input_prompt.get('main_prompt'), 
            input_prompt.get('user_prompt'), 
            input_prompt.get('few_shot_examples'),
            cache=True
        )

        return (system_msg, messages)
    
    def run_batch(self,
                  batch_messages: List[Dict[str, Any]],
                  batch_path: str,
                  experiment_name: str,
                  ):
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
        
        raise NotImplementedError("Batch processing is not implemented for Gemini API.")
        #TODO: Implement batch processing for Gemini API
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        import json

        client = self._get_client()

        # Overwrite the batch input file if it exists
        with open(batch_path, "w") as f:
            f.write("")
        # prepare batch input
        requests = []
        for batch in batch_messages:
            # get the request id and experiment number
            custom_id = batch['batch_id'] # batch id with the request and experiment number 
            system_msg, messages = batch['messages']
            n = batch.get('n', 1)  # Default to 1 if not provided
            params = {
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": messages
                }
            # Add top_p if it's set
            if self.top_p is not None:
                params["top_p"] = self.top_p

            # Add system message if provided
            if system_msg is not None:
                params["system"] = system_msg
            
            if n == 1:
                # Create a single request
                request = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        **params
                    )
                )
                requests.append(request)

                # write the batch message to a jsonl file.
                with open(batch_path, "a") as f:
                    f.write(json.dumps({"custom_id": custom_id, "request": request}) + "\n")

            # If n > 1, create multiple requests with unique custom IDs
            else:
                for i in range(n):
                    request = Request(
                        custom_id=custom_id + f"-{i}",
                        params=MessageCreateParamsNonStreaming(
                            **params
                        )
                    )
                    requests.append(request)

                    # write the batch message to a jsonl file.
                    with open(batch_path, "a") as f:
                        f.write(json.dumps({"custom_id": custom_id + f"-{i}", "request": request}) + "\n")

        message_batch = client.messages.batches.create(requests=requests,)
        
        # Define the batch input file ID
        self.batch_input_file_id = message_batch.id
        # Define the batch ID
        self.batch_id = message_batch.id
        print(f"\nBatch input file uploaded and batch created successfully. The batch ID is {self.batch_id}.\n")

    def get_batch_status(self) -> str:
        """
        Get the status of the batch process.

        Returns:
            str: The status of the batch process.
        """
        raise NotImplementedError("Batch processing is not implemented for Gemini API.")
        #TODO: Implement batch processing for Gemini API
        
        client = self._get_client()

        message_batch = client.messages.batches.retrieve(
            self.batch_id,
        )
        print(f"Batch {message_batch.id} processing status is {message_batch.processing_status}")

        if message_batch.processing_status == "succeeded":
            print(f"Batch {message_batch.id} completed successfully.")
        elif message_batch.processing_status == "errored":
            print(f"Batch {message_batch.id} encountered an error.")
        elif message_batch.processing_status == "canceled":
            print(f"Batch {message_batch.id} was canceled.")
        elif message_batch.processing_status == "in_progress":
            print(f"Batch {message_batch.id} is still processing.")
        else:
            print(f"Batch {message_batch.id} has an unknown status: {message_batch.processing_status}")

        return message_batch.processing_status
    
    def get_batch(self, 
                  batch_response_path: str,
                  ) -> Union[None, Dict[str, Any]]:
        """
        Get the batch prompts used in the SQL generation process.

        Returns:
            dict: A dictionary containing the SQL generation prompt and task.
        """
        raise NotImplementedError("Batch processing is not implemented for Gemini API.")
        #TODO: Implement batch processing for Gemini API

        client = self._get_client()
        
        message_batch = client.messages.batches.retrieve(
            self.batch_id,
        )
        if message_batch.processing_status == "in_progress":
            print(f"Batch {message_batch.id} is not completed yet. Status: {message_batch.processing_status}")
            return None
        elif message_batch.processing_status == "errored" or message_batch.processing_status == "canceled":
            raise Exception(f"Batch {message_batch.id} encountered an error or was canceled. Status: {message_batch.processing_status}")
        else:
            # create the batch response file if it does not exist
            with open(batch_response_path, "w") as f:
                f.write("")

            batch_response = {}
            # Stream results file in memory-efficient chunks, processing one at a time
            for result in client.messages.batches.results(
                self.batch_id,
                ):
                with open(batch_response_path, "a") as f:
                    f.write(json.dumps({'custom_id': result.custom_id, 'response': result.result.message.content[0].text, 'model': result.result.message.model, 'usage': dict(result.result.message.usage)}) + "\n")
                
                match result.result.type:
                    case "succeeded":
                        # print(f"Success! {result.custom_id}")
                        # save line to jsonl file
                        # Extract the relevant information from the result
                        batch_response[result.custom_id] = self.get_batch_response_format(result.result.message)
                    case "errored":
                        if result.result.error.type == "invalid_request":
                            # Request body must be fixed before re-sending request
                            print(f"Validation error {result.custom_id}")
                        else:
                            # Request can be retried directly
                            print(f"Server error {result.custom_id}")
                    case "expired":
                        print(f"Request expired {result.custom_id}")

            return batch_response

    def get_batch_response_format(self, chat_completion: Any) -> Dict[str, Any]:
        """
        Extract relevant information from the OpenAI API response.
        
        Args:
            chat_completion: The response from OpenAI API
            
        Returns:
            dict: Formatted response with extracted information
        """ 
        raise NotImplementedError("Batch processing is not implemented for Gemini API.")
        #TODO: Implement batch processing for Gemini API
        
        responses = {}
        for i, response in enumerate(chat_completion.content):
            responses[i] = response.text
        return {'responses': responses,
                'in_toks': chat_completion.usage.input_tokens,
                'out_toks': chat_completion.usage.output_tokens,
                'id': chat_completion.id,
                'created': chat_completion.id,  # Claude API doesn't have a created timestamp like OpenAI
                'model': chat_completion.model,
                }