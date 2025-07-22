from model.llm import LLMModel
from sql_gen.sqlgen import SQLGenerator
from prompts.prompts_pipeline import get_prompt_version

from utils.utils import get_db_schema_prompt, extract_sql
from constants import DifficultyLevel, GenerationMethod
from typing import Union, List, Dict, Any, Tuple

class StepByStepCoTSQLGenerator(SQLGenerator):
    """
    Class to generate SQL queries using a step-by-step approach based on query difficulty.
    For simple queries, a direct generation approach is used.
    For medium and advanced queries, a step-by-step chain of thought approach is used:
        Ask the model to generate a step-by-step plan before generating the SQL query.
    
    The prompts used for generation are different depending on the difficulty class:
    - Medium queries use simpler planning and SQL generation prompts
    - Advanced queries use more detailed planning and SQL generation prompts
    """

    def __init__(
        self,
        model: LLMModel,
        prompt_version: str = "sbscot_v0",
    ):
        """
        Initialize the StepByStepSQLGenerator class.

        Args:
            model (LLMModel): The language model to use for generation.
            tables_list (list): List of tables relevant for the query.
        """
        self.model = model
        # self.tables_list = tables_list
        prompts = get_prompt_version(prompt_version)
        for key, value in prompts.items():
            setattr(self, key, value)
        
    def get_direct_sql_prompt(self, tables_list: list, ) -> str:
        """
        Get the prompt for direct SQL generation.
        Args:
            tables_list (list): List of tables relevant for the query.
        
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

    def get_sql_with_plan_prompt(self, difficulty_class: str, tables_list: list) -> str:
        """
        Get the prompt for SQL generation with a plan, based on query difficulty.
        
        Args:
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            tables_list (list): List of tables relevant for the query.
            
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
                medium_final_instructions=self.medium_sql_gen_instructions,
            )
        elif difficulty_class == DifficultyLevel.ADVANCED:
            # Use advanced difficulty SQL generation prompt
            prompt = self.adv_sql_gen_prompt_format.format(
                adv_generation_task=self.adv_sql_gen_task,
                adv_query_cntx=self.adv_sql_gen_context,
                db_schema=db_description,
                adv_final_instructions=self.adv_sql_gen_instructions,
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

    def generate_sql_with_steps(self, query: str, difficulty_class: str, tables_list: List[str],
                                ext_knowledge: Union[str, None] = None, dom_knowledge: Union[str, None] = None, n: int = 1, format_sql: bool = True) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
        """
        Generate SQL using a previously generated step-by-step plan, based on query difficulty.
                
        Args:
            query (str): The original query.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            tables_list (List[str]): List of tables relevant for the query.
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            n (int): The number of completions to generate.
            
        Returns:
            tuple: A tuple containing the generated SQL and the model's response.
        """
        # Get the SQL with plan prompt for the specific difficulty class
        sql_with_plan_prompt = self.get_sql_with_plan_prompt(difficulty_class=difficulty_class, tables_list=tables_list)
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
        responses = model_response.get('responses', '')
        plan = {}
        generated_sql = {}
        for k, plan_sql in responses.items():
            # Format the SQL if requested
            plan[k] = plan_sql
            generated_sql[k] = extract_sql(plan_sql, format_sql=format_sql)
        
        # Return the SQL and the model's response
        return generated_sql, plan, model_response
    
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
            tables_list (List[str]): List of tables relevant for the query.
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
            
            sql, plan, sql_response = self.generate_sql_with_steps(query, difficulty_class=difficulty_class, tables_list=tables_list,
                                                             ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, n=n)

            # Return the SQL and details
            return sql, {
                "difficulty_class": difficulty_class,
                "plan": plan,
                "sql_response": sql_response,
            }
        else:
            # Invalid difficulty class
            raise ValueError(
                f"Invalid difficulty class: {difficulty_class}. Expected one of: {', '.join(DifficultyLevel.get_valid_levels())}"
            )
    
    def return_batch(self, query: str, tables_list: List[str], difficulty_class: str,
                         ext_knowledge: Union[str, None], dom_knowledge: Union[str, None], ) -> List[Dict[str, str]]:
        """
        Returns the model input for batching requests for SQL generation with a plan.
        
        Args:
            query (str): The original query.
            tables_list (List[str]): List of tables relevant for the query.
            difficulty_class (str): The difficulty class of the query (medium or advanced).
            ext_knowledge (Union[str, None]): External knowledge to guide SQL generation.
            dom_knowledge (Union[str, None]): Domain-specific knowledge to guide SQL generation.
            
        Returns:
            dict: The model input dictionary containing the main prompt and user prompt.
        """
        if difficulty_class == DifficultyLevel.SIMPLE:
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

        # Check if the plan is None
        elif difficulty_class in [DifficultyLevel.MEDIUM, DifficultyLevel.ADVANCED]:
            # Get the SQL with plan prompt for the specific difficulty class
            sql_with_plan_prompt = self.get_sql_with_plan_prompt(difficulty_class=difficulty_class, tables_list=tables_list)
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
        else:
            # Invalid difficulty class
            raise ValueError(
                f"Invalid difficulty class: {difficulty_class}. Expected one of: {', '.join(DifficultyLevel.get_valid_levels())}"
            )
        # Return the batch for the model
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
            return self.get_sql_with_plan_prompt(difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list)
        elif difficulty_class == DifficultyLevel.ADVANCED:
            return self.get_sql_with_plan_prompt(difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list)
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
            # SQL generation with step-by-step chain of thought
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
            "medium_sql_with_plan_prompt": self.get_sql_with_plan_prompt(difficulty_class=DifficultyLevel.MEDIUM, tables_list=tables_list),
            "adv_sql_with_plan_prompt": self.get_sql_with_plan_prompt(difficulty_class=DifficultyLevel.ADVANCED, tables_list=tables_list),
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