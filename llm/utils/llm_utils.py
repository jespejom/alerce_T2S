import os
from model.llm import LLMModel
from model.gpt import GPTModel
from model.claude import ClaudeModel
from model.vllm import vLLMModel
import tiktoken
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sql_model(model_name: str, **kwargs) -> LLMModel:
    """
    Load and configure a language model based on the model name.
    
    Args:
        model_name (str): Name of the model to use (e.g., 'gpt-4', 'claude-3-haiku', etc.)
        **kwargs: Additional model configuration parameters
            temperature (float): Controls randomness in generation
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Nucleus sampling parameter
            
    Returns:
        LLMModel: The configured language model instance
    """
    temperature = kwargs.get('t', 0.0)
    max_tokens = kwargs.get('max_new_tokens', 4000)
    top_p = kwargs.get('top_p', None)
    
    logger.info(f"Loading model {model_name} with temperature={temperature}, max_tokens={max_tokens}, top_p={top_p}")
    
    # Select the appropriate model class based on the model name prefix
    if model_name.startswith('gpt'):
        # set openai api key
        import openai
        if 'OPENAI_API_KEY' not in os.environ:
            logger.warning("OPENAI_API_KEY not found in environment variables. Setting it now.")
            os.environ['OPENAI_API_KEY'] = input("Please enter your OpenAI API key: ")
        else:
            logger.info("OPENAI_API_KEY found in environment variables.")
        return GPTModel(model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    elif model_name.startswith('claude'):
        # set anthropic api key
        import anthropic
        if 'ANTHROPIC_API_KEY' not in os.environ:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables. Setting it now.")
            os.environ['ANTHROPIC_API_KEY'] = input("Please enter your Anthropic API key: ")
        else:
            logger.info("ANTHROPIC_API_KEY found in environment variables.")

        return ClaudeModel(model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    else:
        # raise ValueError(f"Unsupported model: {model_name}")
        return vLLMModel(model_name, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
    

### Get number of tokens
def get_num_tok(text, encoding_name="cl100k_base"):
  ### Token Counter
  # https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request
  # Encoding name	OpenAI models:
  # - cl100k_base: gpt-4, gpt-3.5-turbo, text-embedding-ada-002
  # - p50k_base:	Codex models, text-davinci-002, text-davinci-003
  # - r50k_base (gpt2):	GPT-3 models like davinci
  enc = tiktoken.get_encoding(encoding_name)
  return len(enc.encode(text))

# OpenAI and Claude model prices - Last Update 24/04/2025
# per 1k tokens
models_info =  [{'model':'gpt-3.5-turbo', 'inp_price': 0.0005 , 'out_price': 0.0015},
                {'model':'gpt-3.5-turbo-0125', 'inp_price': 0.0005, 'out_price': 0.0015},
                {'model':'gpt-3.5-turbo-1106', 'inp_price': 0.001, 'out_price': 0.002},
                {'model':'gpt-3.5-turbo-0613', 'inp_price': 0.0015, 'out_price': 0.002},
                {'model':'gpt-3.5-turbo-16k-0613', 'inp_price': 0.003, 'out_price': 0.004},
                {'model':'gpt-3.5-instruct', 'inp_price': 0.0015, 'out_price': 0.002},
                {'model':'gpt-3.5-turbo-16k', 'inp_price': 0.003, 'out_price': 0.004},
                {'model':'gpt-4o', 'inp_price': 0.005, 'out_price': 0.015},
                {'model':'gpt-4o-2024-05-13', 'inp_price': 0.005, 'out_price': 0.015},
                {'model':'gpt-4o-2024-08-06', 'inp_price': 0.0025, 'out_price': 0.01},
                {'model':'gpt-4o-2024-11-20', 'inp_price': 0.0025, 'out_price': 0.01},
                {'model':'gpt-4o-mini', 'inp_price': 0.000150, 'out_price': 0.000600},
                {'model':'gpt-4o-mini-2024-07-18', 'inp_price': 0.000150, 'out_price': 0.000600},
                {'model':'gpt-4-turbo', 'inp_price': 0.01, 'out_price': 0.03},
                {'model':'gpt-4-1106-preview', 'inp_price': 0.01, 'out_price': 0.03},
                {'model':'gpt-4-0125-preview', 'inp_price': 0.01, 'out_price': 0.03},
                {'model':'gpt-4', 'inp_price': 0.03, 'out_price': 0.06},
                {'model':'gpt-4-32k', 'inp_price': 0.06, 'out_price': 0.12},
                {'model':'gpt-4.1-2025-04-14', 'inp_price': 0.002, 'out_price': 0.008},
                {'model':'text-davinci-003', 'inp_price': 0.02, 'out_price': 0.02},
                {'model':'claude-3-7-sonnet-20250219', 'inp_price': 0.003, 'out_price': 0.015},
                {'model':'claude-3-5-sonnet-20240620', 'inp_price': 0.003, 'out_price': 0.015},
                {'model':'claude-3-opus-20240229', 'inp_price': 0.015, 'out_price': 0.075},
                {'model':'claude-3-haiku-20240307', 'inp_price': 0.0025, 'out_price': 0.00125},]

### Calculate price a paragraph of text
def calc_price(input_text,model='gpt-3.5-turbo', output_tokens=0, inp_tokens=0):
  '''Price Calculator for GPT-Models
  input_text (str): text to calculate price
  model (str): model name
  output_tokens (int): Number of tokens from output text
  inp_tokens (int): Number of tokens from input text
  
  return: price: price for the input text
  '''

  # Number of tokens from input text
  if inp_tokens==0:
    if 'gpt' in model or 'text' in model:
      inp_tokens = get_num_tok(input_text, encoding_name="cl100k_base")
    elif 'claude' in model:
      import anthropic

      vo = anthropic.Client()

      inp_tokens = vo.count_tokens([input_text])
    else:
      raise Exception('Model not found')

  try:
    for i in models_info:
      if i['model'] == model:
        # price: (inp_tokens*input_tokens) + (inp_tokens*output_tokens) per 1k
        return (inp_tokens*i['inp_price'])/1000 + (output_tokens*i['out_price'])/1000  
  
  except Exception as e:
    raise Exception('Model not found')
  


