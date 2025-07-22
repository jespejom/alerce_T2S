from model.llm import LLMModel
from sql_gen.sqlgen import SQLGenerator
from prompts.prompts_pipeline import get_prompt_version

from utils.utils import get_db_schema_prompt, extract_sql
from constants import DifficultyLevel, GenerationMethod
from typing import Union, List, Dict, Any, Tuple

class StepByStepSQLGenerator(SQLGenerator):
    """
    Class to generate SQL queries using a step-by-step approach based on query difficulty.
    For simple queries, a direct generation approach is used.
    For medium and advanced queries, a two-step approach is used:
    1. First generate a step-by-step plan
    2. Then use the plan to generate the SQL query
    
    The prompts used for generation are different depending on the difficulty class:
    - Medium queries use simpler planning and SQL generation prompts
    - Advanced queries use more detailed planning and SQL generation prompts
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str = "sbs_v0",
        # tables_list: list,
        # db_schema: dict = schema_all_cntxV2,
        # Direct SQL generation parameters
        # sql_gen_prompt_format: str = prompt_direct_gen_v0,
        # sql_gen_task: str = prompt_gen_task_v0,
        # sql_gen_context: str = prompt_gen_context_v0 + q3c_info,
        # sql_gen_final_instructions: str = final_instructions_sql_gen_v0,
        # # Medium difficulty planning parameters
        # medium_plan_prompt_format: str = medium_decomp_prompt, 
        # medium_plan_task: str = medium_decomp_task_v3,
        # medium_plan_context: str = prompt_gen_context_v0 + q3c_info,
        # medium_plan_instructions: str = medium_query_instructions_1_v2,
        # # Advanced difficulty planning parameters
        # adv_plan_prompt_format: str = adv_decomp_prompt,
        # adv_plan_task: str = adv_decomp_task_v3,
        # adv_plan_context: str = prompt_gen_context_v0 + q3c_info,
        # adv_plan_instructions: str = adv_query_instructions_1_v3,
        # # Medium difficulty SQL generation with plan parameters
        # medium_sql_gen_prompt_format: str = medium_sql_gen_prompt,
        # medium_sql_gen_task: str = prompt_gen_task_v0,
        # medium_sql_gen_context: str = prompt_gen_context_v0 + q3c_info,
        # medium_sql_gen_instructions: str = final_instructions_sql_gen_v0,
        # # Advanced difficulty SQL generation with plan parameters
        # adv_sql_gen_prompt_format: str = adv_sql_gen_prompt,
        # adv_sql_gen_task: str = prompt_gen_task_v0,
        # adv_sql_gen_context: str = prompt_gen_context_v0 + q3c_info,
        # adv_sql_gen_instructions: str = final_instructions_sql_gen_v0,
    ):
        """
        Initialize the StepByStepSQLGenerator class.

        Args:
            model (LLMModel): The language model to use for generation.
            tables_list (list): List of tables relevant for the query.

            # Prompt parameters
            db_schema (dict): The database schema to use for generation.            
            ## Direct SQL parameters
            sql_gen_prompt_format (str): Template for direct SQL generation.
            sql_gen_task (str): Task description for direct SQL generation.
            sql_gen_context (str): Context for direct SQL generation.
            sql_gen_final_instructions (str): Final instructions for direct SQL generation.
            ## Medium difficulty planning parameters
            medium_plan_prompt_format (str): Template for medium difficulty planning.
            medium_plan_task (str): Task description for medium difficulty planning.
            medium_plan_instructions (str): Instructions for medium difficulty planning.
            ## Advanced difficulty planning parameters
            adv_plan_prompt_format (str): Template for advanced difficulty planning.
            adv_plan_task (str): Task description for advanced difficulty planning.
            adv_plan_instructions (str): Instructions for advanced difficulty planning.
            ## Medium difficulty SQL generation parameters
            medium_sql_gen_prompt_format (str): Template for medium difficulty SQL generation.
            medium_sql_gen_task (str): Task description for medium difficulty SQL generation.
            medium_sql_gen_instructions (str): Instructions for medium difficulty SQL generation.
            ## Advanced difficulty SQL generation parameters
            adv_sql_gen_prompt_format (str): Template for advanced difficulty SQL generation.
            adv_sql_gen_task (str): Task description for advanced difficulty SQL generation.
            adv_sql_gen_instructions (str): Instructions for advanced difficulty SQL generation.
            
        """
        self.model = model
        # self.tables_list = tables_list
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)

        # self.db_schema = db_schema
        
        # Direct SQL generation parameters
        # self.sql_gen_prompt_format = sql_gen_prompt_format
        # self.sql_gen_task = sql_gen_task
        # self.sql_gen_context = sql_gen_context
        # self.sql_gen_final_instructions = sql_gen_final_instructions
        
        # # Medium difficulty planning parameters
        # self.medium_plan_prompt_format = medium_plan_prompt_format
        # self.medium_plan_task = medium_plan_task
        # self.medium_plan_context = medium_plan_context
        # self.medium_plan_instructions = medium_plan_instructions
        
        # # Advanced difficulty planning parameters
        # self.adv_plan_prompt_format = adv_plan_prompt_format
        # self.adv_plan_task = adv_plan_task
        # self.adv_plan_context = adv_plan_context
        # self.adv_plan_instructions = adv_plan_instructions
        
        # # Medium difficulty SQL generation with plan parameters
        # self.medium_sql_gen_prompt_format = medium_sql_gen_prompt_format
        # self.medium_sql_gen_task = medium_sql_gen_task
        # self.medium_sql_gen_context = medium_sql_gen_context
        # self.medium_sql_gen_instructions = medium_sql_gen_instructions
        
        # # Advanced difficulty SQL generation with plan parameters
        # self.adv_sql_gen_prompt_format = adv_sql_gen_prompt_format
        # self.adv_sql_gen_task = adv_sql_gen_task
        # self.adv_sql_gen_context = adv_sql_gen_context
        # self.adv_sql_gen_instructions = adv_sql_gen_instructions
        
    def get_direct_sql_prompt(self, 
                              tables_list: list,
                              ) -> str:
        """
        Get the prompt for direct SQL generation.
        
        Returns:
            str: The formatted prompt for direct SQL generation.
        """
        db_description = get_db_schema_prompt(self.db_schema, tables_list)

        prompt = self.sql_gen_prompt_format.format(
            general_task=self.sql_gen_task,
            general_context=self.sql_gen_context,
            db_schema=db_description,
            final_instructions=self.sql_gen_final_instructions,
        )
        
        return prompt

    def get_plan_gen_prompt(self, difficulty_class: str, tables_list: list) -> str:
        """
        Get the prompt for the step-by-step plan generation, based on query difficulty.
        
        Args:
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            
        Returns:
            str: The formatted prompt for plan generation.
        """
        db_description = get_db_schema_prompt(self.db_schema, tables_list)
        
        if difficulty_class == DifficultyLevel.MEDIUM:
            # Use medium difficulty planning prompt
            prompt = self.medium_plan_prompt_format.format(
                medium_decomp_task=self.medium_plan_task,
                medium_query_cntx=self.medium_plan_context,
                db_schema=db_description,
                medium_final_instructions=self.medium_plan_instructions,
            )
        elif difficulty_class == DifficultyLevel.ADVANCED:
            # Use advanced difficulty planning prompt
            prompt = self.adv_plan_prompt_format.format(
                adv_decomp_task=self.adv_plan_task,
                adv_query_cntx=self.adv_plan_context,
                db_schema=db_description,
                adv_final_instructions=self.adv_plan_instructions,
            )
        else:
            raise ValueError(f"Invalid difficulty class for plan generation: '{difficulty_class}'")
        
        return prompt

    def get_sql_with_plan_prompt(self, plan: str, difficulty_class: str, tables_list: list) -> str:
        """
        Get the prompt for SQL generation with a plan, based on query difficulty.
        
        Args:
            plan (str): The plan to include in the prompt.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            
        Returns:
            str: The formatted prompt for SQL generation with a plan.
        """
        db_description = get_db_schema_prompt(self.db_schema, tables_list)

        if difficulty_class == DifficultyLevel.MEDIUM:
            # Use medium difficulty SQL generation prompt
            prompt = self.medium_sql_gen_prompt_format.format(
                medium_generation_task=self.medium_sql_gen_task,
                medium_query_cntx=self.medium_sql_gen_context,
                db_schema=db_description,
                decomp_plan=plan,
                medium_final_instructions=self.medium_sql_gen_instructions,
                final_instructions=self.medium_sql_gen_instructions,
            )
        elif difficulty_class == DifficultyLevel.ADVANCED:
            # Use advanced difficulty SQL generation prompt
            prompt = self.adv_sql_gen_prompt_format.format(
                adv_generation_task=self.adv_sql_gen_task,
                adv_query_cntx=self.adv_sql_gen_context,
                db_schema=db_description,
                decomp_plan=plan,
                adv_final_instructions=self.adv_sql_gen_instructions,
                final_instructions=self.adv_sql_gen_instructions,
            )
        else:
            raise ValueError(f"Invalid difficulty class for SQL generation: {difficulty_class}")
        
        return prompt

    def generate_sql_directly(self, query: str, tables_list: List[str],
                              ext_knowledge: str, dom_knowledge: str, n: int = 1, format_sql: bool = True) -> tuple:
        """
        Generate SQL directly from the query.
        
        Args:
            query (str): The query to convert to SQL.
            ext_knowledge (str): External knowledge to consider.
            dom_knowledge (str): Domain knowledge to consider.
            n (int): The number of completions to generate.
            format_sql (bool): Whether to format the SQL query or not.

        Returns:
            tuple: A tuple containing the generated SQL and the model's response.
        """
        # Get the direct SQL generation prompt
        direct_prompt = self.get_direct_sql_prompt(tables_list=tables_list)
        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
        
        # Format the messages for the model
        mssgs_input = {
            # "main_prompt": direct_prompt+"\n"+extra_knowledge,
            # "user_prompt": query,
            "main_prompt": direct_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,
        }
        
        # Generate the SQL
        pred_output = self.model.generate(mssgs_input, n=n)
        
        # Extract the generated SQL query from the model's response
        generated_sql = pred_output.get('responses', '')
        for k in generated_sql.keys():
            # Extract SQL from each response
            generated_sql[k] = generated_sql[k]
            # Format the SQL if requested
            generated_sql[k] = extract_sql(generated_sql[k], format_sql=format_sql)
        return generated_sql, pred_output

    def generate_steps_plan(self, query: str, difficulty_class: str, tables_list: List[str],
                            ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None, n: int = 1) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        Generate a step-by-step plan for the given query based on its difficulty class.
        This plan will be used to guide the SQL generation process.
        
        Args:
            query (str): The query to create a plan for.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            ext_knowledge (str): External knowledge to consider.
            dom_knowledge (str): Domain knowledge to consider.
            n (int): The number of completions to generate.
            
        Returns:
            tuple: A tuple containing the generated plan and the model's response.
        """
        # Get the plan generation prompt for the specific difficulty class
        plan_prompt = self.get_plan_gen_prompt(difficulty_class=difficulty_class, tables_list=tables_list)
        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
        
        # Format the messages for the model
        mssgs_input = {
            # "main_prompt": plan_prompt+"\n"+extra_knowledge,
            # "user_prompt": query,
            "main_prompt": plan_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,
        }
        
        # Generate the plan
        pred_output = self.model.generate(mssgs_input, n)
        
        generated_plan = pred_output.get('responses', '')
        for k in generated_plan.keys():
            # Extract the plan from each response
            generated_plan[k] = generated_plan[k]
        # Return the plan and the model's response
        return generated_plan, pred_output

    def generate_sql_with_steps(self, query: str, plan: str, difficulty_class: str, tables_list: List[str],
                                ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None, n: int = 1, format_sql: bool = True) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        Generate SQL using a previously generated step-by-step plan, based on query difficulty.
                
        Args:
            query (str): The original query.
            plan (str): The plan to guide SQL generation.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            n (int): The number of completions to generate.
            
        Returns:
            tuple: A tuple containing the generated SQL and the model's response.
        """
        # Get the SQL with plan prompt for the specific difficulty class
        sql_with_plan_prompt = self.get_sql_with_plan_prompt(plan=plan, difficulty_class=difficulty_class, tables_list=tables_list)
        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
        
        # Format the messages for the model
        mssgs_input = {
            # "main_prompt": sql_with_plan_prompt+"\n"+extra_knowledge,
            # "user_prompt": query,
            "main_prompt": sql_with_plan_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,
        }
        
        # Generate the SQL
        model_response = self.model.generate(mssgs_input, n=n)
        
        # Extract the SQL from the response
        generated_sql = model_response.get('responses', '')
        for k in generated_sql.keys():
            # Extract SQL from each response
            generated_sql[k] = generated_sql[k]
            # Format the SQL if requested
            generated_sql[k] = extract_sql(generated_sql[k], format_sql=format_sql)
        
        # Return the SQL and the model's response
        return generated_sql, model_response
    
    def generate_sql(self, query: str, difficulty_class: str, tables_list: List[str],
                     ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None,
                     n: int = 1, format_sql=True) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        Generate SQL based on the difficulty class of the query.
        For simple queries, generate SQL directly.
        For medium and advanced queries, first generate a plan, then generate SQL using the plan.
        
        Args:
            query (str): The query to convert to SQL.
            difficulty_class (str): The difficulty class of the query (simple, medium, advanced).
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            n (int): The number of completions to generate.
            format_sql (bool): Whether to format the SQL query or not. Defaults to True.
            
        Returns:
            tuple: A tuple containing the generated SQL and a dictionary with details of the generation process.
        """
        # Normalize the difficulty class
        difficulty_class = difficulty_class.lower()
        
        # Check the difficulty class and use the appropriate method
        if difficulty_class == DifficultyLevel.SIMPLE:
            # For simple queries, generate SQL directly
            sql, model_response = self.generate_sql_directly(query, tables_list=tables_list, 
                                                             ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, 
                                                             n=n, format_sql=format_sql)

            # Return the SQL and details
            return sql, {
                "difficulty_class": difficulty_class,
                "sql": sql,
                "sql_response": model_response,
                "plan": None,
                "plan_response": None,
            }
        elif difficulty_class in [DifficultyLevel.MEDIUM, DifficultyLevel.ADVANCED]:
            # For medium and advanced queries, use the two-step approach
            # with difficulty-specific prompts
            
            # First, generate a plan using the appropriate difficulty-specific prompt
            plan, plan_response = self.generate_steps_plan(query, difficulty_class=difficulty_class, tables_list=tables_list,
                                                           ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, n=n)
            sql = {}
            final_sql_response = {}
            for p in plan.keys():
                # generate SQL using the plan and the appropriate difficulty-specific prompt
                generated_sql, sql_response = self.generate_sql_with_steps(query, plan=plan[p], difficulty_class=difficulty_class, tables_list=tables_list,
                                                             ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, n=1)
                
                sql[p] = generated_sql[0]
                final_sql_response[p] = sql_response

            # Return the SQL and details
            return sql, {
                "difficulty_class": difficulty_class,
                "plan": plan,
                "plan_response": plan_response,
                "sql_response": final_sql_response,
            }
        else:
            # Invalid difficulty class
            raise ValueError(
                f"Invalid difficulty class: {difficulty_class}. Expected one of: {', '.join(DifficultyLevel.get_valid_levels())}"
            )
        
    def return_batch_plan(self, query: str, difficulty_class: str, tables_list: List[str],
                          ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None,
                          ) -> List[Dict[str, str]]:
        """
        Returns the model input for batching requests for plan generation.
        
        Args:
            query (str): The query to create a plan for.
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            
        Returns:
            dict: The model input dictionary containing the main prompt and user prompt.
        """
        # Get the plan generation prompt for the specific difficulty class
        plan_prompt = self.get_plan_gen_prompt(difficulty_class=difficulty_class, tables_list=tables_list)
        # Prepare extra knowledge
        extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
        extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
        extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
        
        # Format the messages for the model
        mssgs_input = {
            # "main_prompt": plan_prompt+"\n"+extra_knowledge,
            # "user_prompt": query,
            "main_prompt": plan_prompt,
            "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
            "few_shot_examples": None,
        }
        
        batch = self.model.return_batch(mssgs_input)
        return batch
    
    def return_batch_sql(self, query: str, tables_list: List[str], difficulty_class: str,
                         ext_knowledge: Union[str, None], dom_knowledge: Union[str, None], 
                         plan: Union[str, None], ) -> List[Dict[str, str]]:
        """
        Returns the model input for batching requests for SQL generation with a plan.
        
        Args:
            query (str): The original query.
            tables_list (List[str]): List of tables relevant for the query.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            plan (Union[str, None]): The plan to guide SQL generation.
            
        Returns:
            dict: The model input dictionary containing the main prompt and user prompt.
        """

        # Check if the plan is None
        if plan:
            # Get the SQL with plan prompt for the specific difficulty class
            sql_with_plan_prompt = self.get_sql_with_plan_prompt(plan, difficulty_class=difficulty_class, tables_list=tables_list)
            # Prepare extra knowledge
            extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
            extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
            extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
            
            # Format the messages for the model
            mssgs_input = {
                # "main_prompt": sql_with_plan_prompt+"\n"+extra_knowledge,
                # "user_prompt": query,
                "main_prompt": sql_with_plan_prompt,
                "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
                "few_shot_examples": None,
            }

            batch = self.model.return_batch(mssgs_input)
        # if plan is None, the query is classified as simple
        # and the direct SQL generation prompt is used
        else:
            # Get the direct SQL generation prompt
            direct_prompt = self.get_direct_sql_prompt(tables_list=tables_list)
            # Prepare extra knowledge
            extra_knowledge = "# Important Information for the query\n" if ext_knowledge or dom_knowledge else ""
            extra_knowledge += "External Knowledge: "+ str(ext_knowledge) if ext_knowledge else ""
            extra_knowledge += "\nDomain Knowledge: "+ str(dom_knowledge) if dom_knowledge else ""
            
            # Format the messages for the model
            mssgs_input = {
                # "main_prompt": direct_prompt+"\n"+extra_knowledge,
                # "user_prompt": query,
                "main_prompt": direct_prompt,
                "user_prompt": extra_knowledge + f"\n # User Request: ''{query}''",
                "few_shot_examples": None,
            }
            
            batch = self.model.return_batch(mssgs_input)        
        return batch

    def get_prompt(self, difficulty_class: str, tables_list: List[str] = [""]) -> str:
        """
        Get the prompt for SQL generation based on the difficulty class.
        
        Args:
            difficulty_class (str): The difficulty class of the query (simple, medium, advanced).
            tables_list (List[str]): List of tables relevant for the query.
            
        Returns:
            str: The formatted prompt for SQL generation.
        """
        if difficulty_class == DifficultyLevel.SIMPLE:
            return self.get_direct_sql_prompt(tables_list=tables_list)
        elif difficulty_class == DifficultyLevel.MEDIUM:
            return self.get_plan_gen_prompt(difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list) \
                   + "\n\n SQL Query Generation Prompt: \n" + self.get_sql_with_plan_prompt(plan="[plan]", difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list)
        elif difficulty_class == DifficultyLevel.ADVANCED:
            return self.get_plan_gen_prompt(difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list) \
                   + "\n\n SQL Query Generation Prompt: \n" + self.get_sql_with_plan_prompt(plan="[plan]", difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list)
        else:
            raise ValueError(f"Invalid difficulty class: {difficulty_class}")
        
    def get_prompts(self, tables_list: List[str] = [""]) -> Dict[str, Any]:
        """
        Get all the prompts used in the SQL generation process.
        
        Returns:
            dict: A dictionary containing all the prompts.
        """

        return {
            "db_schema": self.db_schema,
            # Simple SQL generation
            "simple_format": self.sql_gen_prompt_format,
            "simple_task": self.sql_gen_task,
            "simple_context": self.sql_gen_context,
            "simple_final_instructions": self.sql_gen_final_instructions,
            # plan 
            "med_step_plan_format": self.medium_plan_prompt_format,
            "med_step_plan_task": self.medium_plan_task,
            "med_step_plan_context": self.medium_plan_context,
            "med_step_plan_final_instructions": self.medium_plan_instructions,
            "adv_step_plan_format": self.adv_plan_prompt_format,
            "adv_step_plan_task": self.adv_plan_task,
            "adv_step_plan_context": self.adv_plan_context,
            "adv_step_plan_final_instructions": self.adv_plan_instructions,
            # SQL generation with plan
            "med_sql_gen_format": self.medium_sql_gen_prompt_format,
            "med_sql_gen_task": self.medium_sql_gen_task,
            "med_sql_gen_context": self.medium_sql_gen_context,
            "med_sql_gen_final_instructions": self.medium_sql_gen_instructions,
            "adv_sql_gen_format": self.adv_sql_gen_prompt_format,
            "adv_sql_gen_task": self.adv_sql_gen_task,
            "adv_sql_gen_context": self.adv_sql_gen_context,
            "adv_sql_gen_final_instructions": self.adv_sql_gen_instructions,
            # Full prompts
            "simple_sql_prompt": self.get_direct_sql_prompt(tables_list=tables_list),
            "medium_plan_prompt": self.get_plan_gen_prompt(difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list),
            "adv_plan_prompt": self.get_plan_gen_prompt(difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list),
            "medium_sql_with_plan_prompt": self.get_sql_with_plan_prompt("[plan]", difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list),
            "adv_sql_with_plan_prompt": self.get_sql_with_plan_prompt("[plan]", difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the StepByStepSQLGenerator object to a dictionary.
        
        Returns:
            dict: A dictionary representation of the StepByStepSQLGenerator object.
        """
        return {
            "model": self.model.to_dict(),
            "db_schema": self.db_schema,
            "sql_gen_prompt_format": self.sql_gen_prompt_format,
            "sql_gen_task": self.sql_gen_task,
            "sql_gen_context": self.sql_gen_context,
            "sql_gen_final_instructions": self.sql_gen_final_instructions,
            "medium_plan_prompt_format": self.medium_plan_prompt_format,
            "medium_plan_task": self.medium_plan_task,
            "medium_plan_context": self.medium_plan_context,
            "medium_plan_instructions": self.medium_plan_instructions,
            "adv_plan_prompt_format": self.adv_plan_prompt_format,
            "adv_plan_task": self.adv_plan_task,
            "adv_plan_context": self.adv_plan_context,
            "adv_plan_instructions": self.adv_plan_instructions,
            "medium_sql_gen_prompt_format": self.medium_sql_gen_prompt_format,
            "medium_sql_gen_task": self.medium_sql_gen_task,
            "medium_sql_gen_context": self.medium_sql_gen_context,
            "medium_sql_gen_instructions": self.medium_sql_gen_instructions,
            "adv_sql_gen_prompt_format": self.adv_sql_gen_prompt_format,
            "adv_sql_gen_task": self.adv_sql_gen_task,
            "adv_sql_gen_context": self.adv_sql_gen_context,
            "adv_sql_gen_instructions": self.adv_sql_gen_instructions
        }
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Populate the StepByStepSQLGenerator object from a dictionary.
        
        Args:
            data (dict): A dictionary representation of the StepByStepSQLGenerator object.
        """
        self.model_dict = LLMModel.from_dict(data.get("model")) if data.get("model") else None
        self.db_schema = data.get("db_schema")
        self.sql_gen_prompt_format = data.get("sql_gen_prompt_format")
        self.sql_gen_task = data.get("sql_gen_task")
        self.sql_gen_context = data.get("sql_gen_context")
        self.sql_gen_final_instructions = data.get("sql_gen_final_instructions")
        self.medium_plan_prompt_format = data.get("medium_plan_prompt_format")
        self.medium_plan_task = data.get("medium_plan_task")
        self.medium_plan_context = data.get("medium_plan_context")
        self.medium_plan_instructions = data.get("medium_plan_instructions")
        self.adv_plan_prompt_format = data.get("adv_plan_prompt_format")
        self.adv_plan_task = data.get("adv_plan_task")
        self.adv_plan_context = data.get("adv_plan_context")
        self.adv_plan_instructions = data.get("adv_plan_instructions")
        self.medium_sql_gen_prompt_format = data.get("medium_sql_gen_prompt_format"),
        self.medium_sql_gen_task = data.get("medium_sql_gen_task"),
        self.medium_sql_gen_context = data.get("medium_sql_gen_context"),
        self.medium_sql_gen_instructions = data.get("medium_sql_gen_instructions"),
        self.adv_sql_gen_prompt_format = data.get("adv_sql_gen_prompt_format"),
        self.adv_sql_gen_task = data.get("adv_sql_gen_task"),
        self.adv_sql_gen_context = data.get("adv_sql_gen_context"),
        self.adv_sql_gen_instructions = data.get("adv_sql_gen_instructions")
        # Ensure the model is initialized
        if not isinstance(self.model_dict, LLMModel):
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
        # Ensure the promts are strings
        if not isinstance(self.sql_gen_context, str):
            raise ValueError("SQL generation context must be a string")
        if not all(isinstance(i, str) for i in [
            self.sql_gen_prompt_format,
            self.sql_gen_task,
            self.sql_gen_context,
            self.sql_gen_final_instructions,
            self.medium_plan_prompt_format,
            self.medium_plan_task,
            self.medium_plan_context,
            self.medium_plan_instructions,
            self.adv_plan_prompt_format,
            self.adv_plan_task,
            self.adv_plan_context,
            self.adv_plan_instructions,
            self.medium_sql_gen_prompt_format,
            self.medium_sql_gen_task,
            self.medium_sql_gen_context,
            self.medium_sql_gen_instructions,
            self.adv_sql_gen_prompt_format,
            self.adv_sql_gen_task,
            self.adv_sql_gen_context,
            self.adv_sql_gen_instructions
        ]):
            raise ValueError("All prompts must be strings")