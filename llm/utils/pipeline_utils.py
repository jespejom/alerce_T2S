
from classification.diff_classification import DifficultyClassification
from model.llm import LLMModel

def load_classification_model(
        model: LLMModel,
        promptV_diff_class: str = "diff_v0",
        ) -> DifficultyClassification:

        diff_class_model = DifficultyClassification(model, prompt_version=promptV_diff_class)
        # TODO: add other classification methods

        return diff_class_model


from sql_gen.sqlgen import SQLGenerator
from sql_gen.direct import DirectSQLGenerator
from sql_gen.step import StepByStepSQLGenerator
from sql_gen.stepcot import StepByStepCoTSQLGenerator
from prompts.prompts_pipeline import dir_prompt_versions, sbs_prompt_versions, sbscot_prompt_versions
def load_sql_generator(
        model: LLMModel,
        promptV_gen: str = "dir_v0",
        sql_gen_method: str = "direct",
        ) -> SQLGenerator:

    if sql_gen_method == "direct":
        if promptV_gen not in dir_prompt_versions.keys():
            raise ValueError(f"Prompt version {promptV_gen} not found in the direct SQL generator versions.")
        return DirectSQLGenerator(model=model, prompt_version=promptV_gen)
    elif sql_gen_method == "step-by-step":
        if promptV_gen not in sbs_prompt_versions.keys():
            raise ValueError(f"Prompt version {promptV_gen} not found in the step-by-step SQL generator versions.")
        return StepByStepSQLGenerator(model=model, prompt_version=promptV_gen)
    elif sql_gen_method == "step-by-step-cot":
        if promptV_gen not in sbscot_prompt_versions.keys():
            raise ValueError(f"Prompt version {promptV_gen} not found in the step-by-step-cot SQL generator versions.")
        return StepByStepCoTSQLGenerator(model=model, prompt_version=promptV_gen)
    else:
        raise ValueError(f"Unknown SQL generation method: {sql_gen_method}")


from prompts.prompts_pipeline import get_prompt_version
from utils.utils import get_db_schema_prompt, extract_sql
def load_prompt(
        promptV_gen = 'dir_v8',
        ):
    """
    Load the prompt for SQL generation based on the specified version.
    Args:
        promptV_gen (str): The version of the prompt to load.
    Returns:
        str: The prompt string for SQL generation.
    """
    prompt_parts = get_prompt_version(promptV_gen)
    
    if 'sl' in promptV_gen:
        # This is a SQL generation prompt
        # The prompt_parts should contain the necessary parts for the SQL generation
        # e.g., db_schema, final_instructions, etc.
        # get the tables schema
        db_description = ""
        if isinstance(prompt_parts.get('db_schema'), dict):
            for table_name, table_info in prompt_parts.get('db_schema').items():
                db_description += table_info + "\n"
        elif isinstance(prompt_parts.get('db_schema'), str):
            db_description = prompt_parts.get('db_schema')
        else:
            raise ValueError("db_schema must be a dict or a str")
        return prompt_parts.get('schema_linking_format').format(
            db_schema=db_description,
            final_instructions=prompt_parts.get('sl_final_instructions'),
        )

    elif 'diff' in promptV_gen:
        # This is a difficulty classification prompt
        # The prompt_parts should contain the necessary parts for the difficulty classification
        # e.g., db_schema, final_instructions_diff, etc.
        # get the tables schema
        return prompt_parts.get('diff_class_format').format(
            db_schema=get_db_schema_prompt(prompt_parts.get('db_schema'), ['tables_list']),
            final_instructions_diff=prompt_parts.get('diff_final_instructions'),
        )
    elif 'dir' in promptV_gen:

        db_description = get_db_schema_prompt(prompt_parts.get('db_schema'), ["tables_list"])

        # Construct the prompt using the format with all required placeholders
        prompt = prompt_parts.get('sql_gen_prompt_format').format(
            general_task=prompt_parts.get('sql_gen_task'),
            general_context=prompt_parts.get('sql_gen_context'),
            db_schema=db_description,
            final_instructions=prompt_parts.get('sql_gen_final_instructions'),
        )

        return prompt + "\n\n [User request]"
    
    # TODO: add other prompt versions, and use input from ipywidgets inputs to visualize by difficulty, sql generatio or planning, and other parameters
    # the prompt formatting is done in the respective classes, so we can just return the prompt parts
    # elif 'sbscot' in promptV_gen:
    # elif 'sbs' in promptV_gen:

    # elif 'sc' in promptV_gen:

    #     # Get the database schema description for the relevant tables
    #     db_description = get_db_schema_prompt(prompt_parts.db_schema, ['tables_list'])
        
    #     # Determine error type from the error message
    #     error_info = "error_info"
    #     # TODO: Extract the error type from the error_info
    #     error_type = "error_type"
    #     # Use the cleaned error message
    #     clean_error = "cleaned_error_message"

    #     # Add context about the error class if available
    #     error_context = ""

    #     # TODO from the interactive widget
    #     if error_type == 'timeout':
    #         prompt_format = prompt_parts.timeout_prompt_format
    #     elif error_type == 'not_exist':
    #         prompt_format = prompt_parts.not_exist_prompt_format
    #     else:  # Default to schema error prompt for other errors
    #         prompt_format = prompt_parts.schema_error_prompt_format

    #     # Format the prompt with the specific details
    #     prompt = prompt_format.format(
    #         Self_correction_task=prompt_parts.general_task,
    #         request="query",
    #         tab_schema=db_description,
    #         sql_query="sql_query",
    #         sql_error=f"{error_context}{clean_error}",
    #         final_instructions=prompt_parts.final_instructions,
    #         context=prompt_parts.general_context
    #     )
        