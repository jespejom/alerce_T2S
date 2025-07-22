from typing import Dict, List, Tuple, Any, Optional, Union
import os
import time
import json
import pandas as pd
import logging

# Import model and component classes
from model.llm import LLMModel
from utils.utils import load_dataset
from utils.pipeline_utils import load_sql_generator, load_classification_model
from utils.llm_utils import load_sql_model
from utils.eval_utils import join_eval_results, get_evaluation_stats

# Import components for the pipeline
from schema.schema_linking import SchemaLinking
from sql_gen.sqlgen import SQLGenerator
# from sql_gen.direct import DirectSQLGenerator
# from sql_gen.step import StepByStepSQLGenerator
from classification.diff_classification import DifficultyClassification
from selfcorrection.self_correction import SelfCorrection
from evaluation import evaluate_alerce_queries

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from constants import GenerationMethod, DifficultyLevel

def seq_predict_tables(data: pd.DataFrame,
        n_exps: int,
        schema_linking_model: SchemaLinking,
        schlink_experiments: Optional[Dict[str, Any]] = None
        ):
    """
    Function to generate SQL queries in batches.
    This function is a placeholder and should be implemented based on the specific requirements of the project.
    """
    if schlink_experiments is None:
        # Initialize the schlink_experiments dictionary if not provided
        schlink_experiments = {}
    # If the schema linking model is not provided, raise an error
    if 'schema_linking_model' not in schlink_experiments:
        schlink_experiments['schema_linking_model'] = schema_linking_model.to_dict()

    for index, row in data.iterrows():
        request = row['request']
        req_id = str(row['req_id'])

        # if the runs are already predicted, skip
        if req_id in schlink_experiments and len(schlink_experiments[req_id]) >= n_exps:
            logger.info(f"Skipping schema linking for request {req_id} as it already has {len(schlink_experiments[req_id])} experiments.")
            continue
        # if there are less than n_exps, predict the remaining
        elif req_id in schlink_experiments and len(schlink_experiments[req_id]) < n_exps:
            logger.info(f"Continuing schema linking for request {req_id} as it has {len(schlink_experiments[req_id])} experiments, predicting the remaining {n_exps - len(schlink_experiments[req_id])} experiments.")
            for i in range(len(schlink_experiments[req_id]), n_exps):
                pred_tables, schlink_response = schema_linking_model.predict_tables(request, n=1)
                schlink_experiments[req_id][str(i)] = {
                    "pred_tables": pred_tables,
                    "schema_linking_response": schlink_response
                }
        else:
            schlink_experiments[req_id] = {}
            pred_tables, schlink_response = schema_linking_model.predict_tables(request, n=n_exps)
            for i in range(n_exps):
                schlink_experiments[req_id][str(i)] = {
                    "pred_tables": pred_tables[i],
                    "schema_linking_response": schlink_response
                }

    return schlink_experiments

def seq_difficulty_classification(
        data: pd.DataFrame,
        n_exps: int,
        diff_class_model: DifficultyClassification,
        schlink_experiments: Dict[str, Any],
        diff_experiments: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
    """
    Function to classify the difficulty of SQL queries in batches.
    This function is a placeholder and should be implemented based on the specific requirements of the project.
    """
    if diff_experiments is None:
        # Initialize the diff_experiments dictionary if not provided
        diff_experiments = {}
    for index, row in data.iterrows():
        request = row['request']
        req_id = str(row['req_id'])

        # if the runs are already predicted, skip
        if req_id in diff_experiments and len(diff_experiments[req_id]) >= n_exps:
            logger.info(f"Skipping difficulty classification for request {req_id} as it already has {len(diff_experiments[req_id])} experiments.")
            continue
        # if there are less than n_exps, predict the remaining
        elif req_id in diff_experiments and len(diff_experiments[req_id]) < n_exps:
            logger.info(f"Continuing difficulty classification for request {req_id} as it has {len(diff_experiments[req_id])} experiments, predicting the remaining {n_exps - len(diff_experiments[req_id])} experiments.")
            for i in range(len(diff_experiments[req_id]), n_exps):
                difficulty, diff_response = diff_class_model.classify_difficulty(
                    request, tables_list=schlink_experiments[req_id][str(i)]["pred_tables"], n=1
                )
                diff_experiments[req_id][str(i)] = {
                    "pred_diff": difficulty,
                    "diff_response": diff_response
                }
        else:
            diff_experiments[req_id] = {}
            for i in range(n_exps):
                difficulty, diff_response = diff_class_model.classify_difficulty(
                    request, tables_list=schlink_experiments[req_id][str(i)]["pred_tables"], n=1
                )
                diff_experiments[req_id][str(i)] = {
                    "pred_diff": difficulty,
                    "diff_response": diff_response
                }
    diff_experiments['difficulty_classification_model'] = diff_class_model.to_dict()
    return diff_experiments

def sequential_generate_sql(
        data: pd.DataFrame,
        n_exps: int,
        sql_gen_method: str,
        sql_gen: SQLGenerator,
        schlink_experiments: Dict[str, Any],
        diff_experiments: Optional[Dict[str, Any]] = None,
        all_experiments: Optional[Dict[str, Any]] = None,
        experiment_settings: Optional[Dict[str, Any]] = None,
        **kwargs
        ):
    """
    Function to generate SQL queries in batches.
    This function is a placeholder and should be implemented based on the specific requirements of the project.
    """
    all_experiments = {} if all_experiments is None else all_experiments
    experiment_settings = {} if experiment_settings is None else experiment_settings
    schema_linking_model = schlink_experiments["schema_linking_model"]
    diff_class_model = None if diff_experiments is None else diff_experiments.get('difficulty_classification_model', None)

    for index, row in data.iterrows():
        request = row['request']
        req_id = str(row['req_id'])
        ext_knowledge = row['external_knowledge']
        dom_knowledge = row['domain_knowledge']
        
        if req_id in all_experiments and len(all_experiments[req_id]) >= n_exps:
            logger.info(f"Skipping SQL generation for request {req_id} as it already has {len(all_experiments[req_id])} experiments.")
            continue

        elif req_id in all_experiments and len(all_experiments[req_id]) < n_exps:
            logger.info(f"Continuing SQL generation for request {req_id} as it has {len(all_experiments[req_id])} experiments, predicting the remaining {n_exps - len(all_experiments[req_id])} experiments.")
            # Generate SQL for the remaining experiments
            for i in range(len(all_experiments[req_id]), n_exps):
                pred_tables = schlink_experiments[req_id][str(i)]["pred_tables"]
                schlink_response = schlink_experiments[req_id][str(i)]["schema_linking_response"]
                if diff_experiments is not None:
                    # If difficulty classification is provided, use it
                    difficulty = diff_experiments[req_id][str(i)]["pred_diff"]
                    diff_response = diff_experiments[req_id][str(i)]["diff_response"]
                else:
                    # If not, set difficulty to None
                    difficulty = None
                    diff_response = None

                experiment_id = f"run_{i}"
                # Generate SQL
                if sql_gen_method == GenerationMethod.DIRECT:
                    sql_query_results, sql_response = sql_gen.generate_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method in [GenerationMethod.STEP_BY_STEP, GenerationMethod.STEP_BY_STEP_COT]:
                    # Generate SQL using step-by-step method
                    sql_query_results, sql_response = sql_gen.generate_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, difficulty_class=difficulty)
                else:
                    raise ValueError(f"Unsupported SQL generation method: {sql_gen_method}")
                
                # Save experiment results for this run
                experiment_results = {
                    "req_id": req_id,
                    "experiment_id": experiment_id,
                    "pred_tables": pred_tables,
                    "schema_linking_response": schlink_response,
                    "sql_query": sql_query_results[0],
                    "sql_response": sql_response,
                    "pred_diff": difficulty,
                    "diff_response": diff_response,
                }
                experiment_pipeline = {
                    "schema_linking": schema_linking_model,
                    "sql_generation": sql_gen.to_dict(),
                    "difficulty_classification": diff_class_model
                }
                
                all_experiments[req_id][i] = experiment_results
                experiment_settings[req_id][i] = experiment_pipeline

        else:
            all_experiments[req_id] = {}
            experiment_settings[req_id] = {}

            logger.info(f"Running experiments for request ID: {req_id}")
            for i in range(n_exps):
                pred_tables = schlink_experiments[req_id][str(i)]["pred_tables"]
                schlink_response = schlink_experiments[req_id][str(i)]["schema_linking_response"]
                if diff_experiments is not None:
                    # If difficulty classification is provided, use it
                    difficulty = diff_experiments[req_id][str(i)]["pred_diff"]
                    diff_response = diff_experiments[req_id][str(i)]["diff_response"]
                else:
                    # If not, set difficulty to None
                    difficulty = None
                    diff_response = None

                experiment_id = f"run_{i}"
                # Generate SQL
                if sql_gen_method == GenerationMethod.DIRECT:
                    sql_query_results, sql_response = sql_gen.generate_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method in [GenerationMethod.STEP_BY_STEP, GenerationMethod.STEP_BY_STEP_COT]:
                    # Generate SQL using step-by-step method
                    sql_query_results, sql_response = sql_gen.generate_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, difficulty_class=difficulty)
                else:
                    raise ValueError(f"Unsupported SQL generation method: {sql_gen_method}")
                
                # Save experiment results for this run
                experiment_results = {
                    "req_id": req_id,
                    "experiment_id": experiment_id,
                    "pred_tables": pred_tables,
                    "schema_linking_response": schlink_response,
                    "sql_query": sql_query_results[0],
                    "sql_response": sql_response,
                    "pred_diff": difficulty,
                    "diff_response": diff_response,
                }
                experiment_pipeline = {
                    "schema_linking": schema_linking_model,
                    "sql_generation": sql_gen.to_dict(),
                    "difficulty_classification": diff_class_model
                }
                
                all_experiments[req_id][i] = experiment_results
                experiment_settings[req_id][i] = experiment_pipeline
                # logger.info(f"Experiment {i + 1}/{n_exps} completed")

    return all_experiments, experiment_settings
    
def batch_generate_sql(
        data: pd.DataFrame,
        exp_name: str,
        exp_dir: str,
        model: LLMModel,
        n_exps: int,
        sql_gen_method: str,
        schlink_experiments: Dict[str, Any],
        sql_gen: SQLGenerator,
        diff_experiments: Optional[Dict[str, Any]] = None,
        all_experiments: Optional[Dict[str, Any]] = None,
        experiment_settings: Optional[Dict[str, Any]] = None,
        **kwargs
        ):
    """
    Function to generate SQL queries in batches.
    This function is a placeholder and should be implemented based on the specific requirements of the project.
    """
    all_experiments = {} if all_experiments is None else all_experiments
    experiment_settings = {} if experiment_settings is None else experiment_settings
    batch_messages = []
    
    schema_linking_model = schlink_experiments["schema_linking_model"]
    diff_class_model = None if diff_experiments is None else diff_experiments.get('difficulty_classification_model', None)
    
    for index, row in data.iterrows():
        request = row['request']
        req_id = str(row['req_id'])
        ext_knowledge = row['external_knowledge']
        dom_knowledge = row['domain_knowledge']
        
        # if the runs are already predicted, skip
        if req_id in all_experiments and len(all_experiments[req_id]) >= n_exps:
            logger.info(f"Skipping SQL generation for request {req_id} as it already has {len(all_experiments[req_id])} experiments.")
            continue
        # if there are less than n_exps, predict the remaining
        elif req_id in all_experiments and len(all_experiments[req_id]) < n_exps:
            logger.info(f"Continuing SQL generation for request {req_id} as it has {len(all_experiments[req_id])} experiments, predicting the remaining {n_exps - len(all_experiments[req_id])} experiments.")
            # Generate SQL for the remaining experiments
            for i in range(len(all_experiments[req_id]), n_exps):
                pred_tables = schlink_experiments[req_id][str(i)]["pred_tables"]
                schlink_response = schlink_experiments[req_id][str(i)]["schema_linking_response"]
                if diff_experiments is not None:
                    # If difficulty classification is provided, use it
                    difficulty = diff_experiments[req_id][str(i)]["pred_diff"]
                    diff_response = diff_experiments[req_id][str(i)]["diff_response"]
                else:
                    # If not, set difficulty to None
                    difficulty = None
                    diff_response = None
                
                # Generate SQL
                if sql_gen_method == GenerationMethod.DIRECT:
                    sql_query_messages = sql_gen.return_batch(request, tables_list=pred_tables, 
                                                              ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method == GenerationMethod.STEP_BY_STEP:
                    # Generate SQL using step-by-step method
                    if difficulty == DifficultyLevel.SIMPLE:
                        sql_query_messages = sql_gen.return_batch_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, plan=None, difficulty_class=difficulty)
                    else:
                        sql_query_messages = sql_gen.return_batch_plan(request, tables_list=pred_tables, difficulty_class=difficulty, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method == GenerationMethod.STEP_BY_STEP_COT:
                    # Generate SQL using step-by-step method with chain of thought
                    sql_query_messages = sql_gen.return_batch(request, tables_list=pred_tables, difficulty_class=difficulty,
                                                              ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                else:
                    raise ValueError(f"Unsupported SQL generation method: {sql_gen_method}")
                
                batch_messages.append({"batch_id": f"{req_id}-{i}", "messages": sql_query_messages, "n": 1})
                
                # Save experiment results for this run
                experiment_id = f"run_{i}"
                experiment_results = {
                    "req_id": req_id,
                    "experiment_id": experiment_id,
                    "pred_tables": pred_tables,
                    "schema_linking_response": schlink_response,
                    "sql_query": sql_query_messages,
                    "sql_response": None,
                    "pred_diff": difficulty,
                    "diff_response": diff_response,
                }
                all_experiments[req_id][str(i)] = experiment_results
        else:
            all_experiments[req_id] = {}
            experiment_settings[req_id] = {}

            for i in range(n_exps):
                pred_tables = schlink_experiments[req_id][str(i)]["pred_tables"]
                schlink_response = schlink_experiments[req_id][str(i)]["schema_linking_response"]
                if diff_experiments is not None:
                    # If difficulty classification is provided, use it
                    difficulty = diff_experiments[req_id][str(i)]["pred_diff"]
                    diff_response = diff_experiments[req_id][str(i)]["diff_response"]
                else:
                    # If not, set difficulty to None
                    difficulty = None
                    diff_response = None
                
                # Generate SQL
                if sql_gen_method == GenerationMethod.DIRECT:
                    sql_query_messages = sql_gen.return_batch(request, tables_list=pred_tables, 
                                                            ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method == GenerationMethod.STEP_BY_STEP:
                    # Generate SQL using step-by-step method
                    # sql_query_results, sql_response = sql_gen.generate_sql(request, difficulty)
                    if difficulty == DifficultyLevel.SIMPLE:
                        sql_query_messages = sql_gen.return_batch_sql(request, tables_list=pred_tables, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, plan=None, difficulty_class=difficulty)
                    else:
                        sql_query_messages = sql_gen.return_batch_plan(request, tables_list=pred_tables, difficulty_class=difficulty, ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                elif sql_gen_method == GenerationMethod.STEP_BY_STEP_COT:
                    # Generate SQL using step-by-step method with chain of thought
                    sql_query_messages = sql_gen.return_batch(request, tables_list=pred_tables, difficulty_class=difficulty,
                                                            ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge)
                else:
                    raise ValueError(f"Unsupported SQL generation method: {sql_gen_method}")
                # batch_messages.append({"batch_id": req_id, "messages": sql_query_messages, "n": n_exps})
                batch_messages.append({"batch_id": f"{req_id}-{i}", "messages": sql_query_messages, "n": 1})
                # Save experiment results for this run
                experiment_id = f"run_{i}"
                experiment_results = {
                    "req_id": req_id,
                    "experiment_id": experiment_id,
                    "pred_tables": pred_tables,
                    "schema_linking_response": schlink_response,
                    "sql_query": sql_query_messages,
                    "sql_response": None,
                    "pred_diff": difficulty,
                    "diff_response": diff_response,
                }
                all_experiments[req_id][str(i)] = experiment_results

    experiment_settings = {
        "schema_linking": schema_linking_model,
        "sql_generation": sql_gen.to_dict(),
        "difficulty_classification": diff_class_model
        }
        # logger.info(f"Experiment {i + 1}/{n_exps} completed")

    if batch_messages == []:
        logger.info("No batch messages to send to the model. Exiting...")
        return all_experiments, experiment_settings
    
    # send the batch messages to the model
    model.run_batch(
        batch_messages,
        os.path.join(exp_dir, "batch_messages.jsonl"),
        exp_name
    )

    # wait for the model to finish
    # set a timeout for the batch response
    logger.info("Waiting for the model to finish processing the batch requests...")
    while True:
        batch_response = model.get_batch(
            os.path.join(exp_dir, "batch_messages_response.jsonl"),
            )
        time.sleep(300) # wait for 5 minutes
        if batch_response is not None:
            break
        else:
            logger.info("Waiting for the model to finish...")

    if sql_gen_method == GenerationMethod.STEP_BY_STEP:
        batch_sbs_messages = []
        # using the step-by-step plan, we need to get the SQL query from the model response
        # for each message in the batch, get the SQL query and save it to the all_experiments dict
        # for batch in batch_response:
        for batch_key in batch_response.keys():
            # get the request id and experiment number
            req_id = batch_key.split("-")[0]
            n_exp = batch_key.split("-")[1]
            for null_exp in batch_response[batch_key]['responses'].keys(): # just one response
                sql_plan = batch_response[batch_key]['responses'][null_exp]
                request = data.loc[data['req_id'] == int(req_id), 'request'].values[0]
                difficulty = all_experiments[req_id][n_exp]["pred_diff"]
                ext_knowledge = data.loc[data['req_id'] == int(req_id), 'external_knowledge'].values[0]
                dom_knowledge = data.loc[data['req_id'] == int(req_id), 'domain_knowledge'].values[0]
                if difficulty == DifficultyLevel.SIMPLE:
                    # add the generated SQL query to the all_experiments dict
                    all_experiments[req_id][n_exp]["sql_query"] = sql_plan
                    all_experiments[req_id][n_exp]["sql_response"] = batch_response[batch_key]
                    all_experiments[req_id][n_exp]["sql_query_plan"] = None
                    all_experiments[req_id][n_exp]["sql_query_plan_response"] = None
                else:
                    # add the SQL prompt to the batch_sbs_messages
                    sql_query_messages = sql_gen.return_batch_sql(request, tables_list=pred_tables, difficulty_class=difficulty,
                                                                  ext_knowledge=ext_knowledge, dom_knowledge=dom_knowledge, plan=sql_plan)
                    batch_sbs_messages.append({"batch_id": f"{req_id}-{n_exp}", "messages": sql_query_messages, "n": 1})
                    
                    all_experiments[req_id][n_exp]["sql_query_plan"] = sql_plan
                    all_experiments[req_id][n_exp]["sql_query_plan_response"] = batch_response[batch_key]

        # send the batch messages to the model
        model.run_batch(
            batch_sbs_messages,
            os.path.join(exp_dir, "batch_sbs_messages.jsonl"),
            exp_name
        )
        # wait for the model to finish
        while True:
            batch_response = model.get_batch(
                os.path.join(exp_dir, "batch_sbs_messages_response.jsonl"),
                )
            time.sleep(300) # wait for 5 minutes
            if batch_response is not None:
                break
            else:
                logger.info("Waiting for the model to finish the step-by-step SQL generation...")
                
    # get the request id and experiment number
    # for each message in the batch, get the result and save it to the all_experiments dict
    for batch_key in batch_response.keys():
    # for batch in batch_response:
        # get the request id and experiment number
        req_id = batch_key.split("-")[0]
        n_exp = batch_key.split("-")[1]
        for null_exp in batch_response[batch_key]['responses'].keys(): # just one response
            # get the SQL query from the model response
            all_experiments[req_id][n_exp]["sql_query"] = batch_response[batch_key]['responses'][null_exp]
            all_experiments[req_id][n_exp]["sql_response"] = batch_response[batch_key]

    return all_experiments, experiment_settings

def run_pipeline(
        data_path: str,
        data_format: str,
        model_name: str,
        save_path: str,
        exp_name: str,
        n_exps: int = 1,
        eval: bool = True,
        batch: bool = False,
        self_correction: bool = True,
        schema_linking_method: bool = True,
        sql_gen_method: str = GenerationMethod.DIRECT,
        promptV_sl: str = "sl_v0",
        promptV_diff_class: str = "diff_v0",
        promptV_gen: str = "dir_v0",
        promptV_sc: str = "sc_v0",
        **kwargs
) -> None:
    """
    Run the text-to-SQL experiment with the given parameters.

    Args:
        data_path (str): The path to the data file.
        data_format (str): The format of the data file (e.g., 'csv', 'json').
        model_name (str): The name of the model to use.
        save_path (str): The base path to save the results.
        exp_name (str): The name of the experiment to run.
        n_exps (int): The number of experiments to run.
        eval (bool): Whether to run evaluation after the experiment.
        self_correction (bool): Whether to apply self-correction to the results.
        schema_linking_method (bool): Whether to use schema linking method.
        sql_gen_method (str): The SQL generation method to use (direct or step-by-step).
        promptV_sl (str): The version of the schema linking prompt to use.
        promptV_diff_class (str): The version of the difficulty classification prompt to use.
        promptV_gen (str): The version of the SQL generation prompt to use.
        **kwargs: Additional keyword arguments for the experiment.
    """
    # Parse the arguments
    args = parse_args()
    
    # Get evaluation and self-correction flags
    eval_flag = eval
    self_correction_flag = self_correction
    
    # Load the data
    data = load_dataset(data_path, data_format)

    # Create the result directory structure: results/model_name/experiment_name/
    model_dir = os.path.basename(model_name)  # Extract just the model name from potential paths
    experiment_dir = os.path.join(save_path, model_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load the model
    model = load_sql_model(model_name, **kwargs)
    if args.model_name_decomp is not None:
        model_decomp = load_sql_model(args.model_name_decomp, **kwargs)
    else:
        model_decomp = model

    # Load the Schema Linking model
    if schema_linking_method:
        # Load the schema linking model
        schema_linking_model = SchemaLinking(model, prompt_version=promptV_sl)
        # Check if the schema linking model has a database schema
        os.makedirs(os.path.join(save_path, model_dir, "schema_linking"), exist_ok=True)
        # If the schema linking was already run, load the predicted tables
        if os.path.exists(os.path.join(save_path, model_dir, "schema_linking", os.path.basename(data_path)+"_"+schema_linking_model.get_prompt_version()+"_predicted_tables.json")):
            # Load the predicted tables from the file
            logger.info(f"Loading existing schema linking predictions from {os.path.join(save_path, model_dir, 'schema_linking', os.path.basename(data_path)+'_'+schema_linking_model.get_prompt_version()+'_predicted_tables.json')}")
            with open(os.path.join(save_path, model_dir, "schema_linking", os.path.basename(data_path)+"_"+schema_linking_model.get_prompt_version()+"_predicted_tables.json"), "r") as f:
                schlink_experiments = json.load(f)
        else:
            # If the schema linking was not run, initialize an empty dictionary
            schlink_experiments = None
        schlink_experiments = seq_predict_tables(data,
                                                 n_exps,
                                                 schema_linking_model,
                                                 schlink_experiments=schlink_experiments)
        # Save the schema linking results
        with open(os.path.join(save_path, model_dir, "schema_linking", os.path.basename(data_path)+"_"+schema_linking_model.get_prompt_version()+"_predicted_tables.json"), "w") as f:
            json.dump(schlink_experiments, f, indent=4)
    else:
        # Use a default schema linking model
        # TODO: Implement a default schema linking model
        raise NotImplementedError("Default schema linking model is not implemented.")
    
    # Load the SQL generation model
    sql_gen = load_sql_generator(
        model=model,
        promptV_gen=promptV_gen,
        sql_gen_method=sql_gen_method
    )
    # Load the difficulty classification model if using step-by-step SQL generation
    if sql_gen_method in [GenerationMethod.STEP_BY_STEP, GenerationMethod.STEP_BY_STEP_COT]:
        diff_class_model = DifficultyClassification(model_decomp, prompt_version=promptV_diff_class)
        # Create the directory for difficulty classification results
        os.makedirs(os.path.join(save_path, model_dir, "difficulty_classification"), exist_ok=True)
        # If the difficulty classification was already run, load the predicted difficulties
        if os.path.exists(os.path.join(save_path, model_dir, "difficulty_classification", os.path.basename(data_path)+"_"+diff_class_model.get_prompt_version()+"_predicted_difficulties.json")):
            # Load the predicted difficulties from the file
            logger.info(f"Loading existing difficulty classification predictions from {os.path.join(save_path, model_dir, 'difficulty_classification', os.path.basename(data_path)+'_'+diff_class_model.get_prompt_version()+'_predicted_difficulties.json')}")
            with open(os.path.join(save_path, model_dir, "difficulty_classification", os.path.basename(data_path)+"_"+diff_class_model.get_prompt_version()+"_predicted_difficulties.json"), "r") as f:
                diff_experiments = json.load(f)
        else:
            # If the difficulty classification was not run, initialize an empty dictionary
            diff_experiments = None
        diff_experiments = seq_difficulty_classification(
            data=data,
            n_exps=n_exps,
            diff_class_model=diff_class_model,
            schlink_experiments=schlink_experiments,
            diff_experiments=diff_experiments
        )
        # Save the difficulty classification results
        with open(os.path.join(save_path, model_dir, "difficulty_classification", os.path.basename(data_path)+"_"+diff_class_model.get_prompt_version()+"_predicted_difficulties.json"), "w") as f:
            json.dump(diff_experiments, f, indent=4)
    else: 
        diff_experiments = None
    
    # Check if experiment file already exists
    if os.path.exists(os.path.join(experiment_dir, exp_name + ".json")):
        logger.info(f"Experiment file already exists at {os.path.join(experiment_dir, exp_name + '.json')}")
        # Load existing experiments
        with open(os.path.join(experiment_dir, exp_name + ".json"), "r") as f:
            all_experiments = json.load(f)
    else:
        all_experiments = None
    # Check if configuration file already exists
    if os.path.exists(os.path.join(experiment_dir, "config.json")):
        logger.info(f"Configuration file already exists at {os.path.join(experiment_dir, 'config.json')}")
        # Load existing configuration
        with open(os.path.join(experiment_dir, "config.json"), "r") as f:
            config = json.load(f)
        experiment_settings = config.get("experiment_settings", {})
    else:
        experiment_settings = None

    # Save configuration with prompt information
    config = vars(args).copy()
    config["model_name"] = model_name
    config["save_path"] = save_path
    config["sql_gen_method"] = sql_gen_method
    config["date_executed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Iterate over the data and run experiments
    if batch:
        # Call batch_generate_sql to process data in batches
        all_experiments, experiment_settings = batch_generate_sql(
            data=data,
            exp_name=exp_name,
            data_path=data_path,
            exp_dir=experiment_dir,
            model=model,
            model_decomp=model_decomp,
            n_exps=n_exps,
            sql_gen_method=sql_gen_method,
            promptV_gen=promptV_gen,
            promptV_diff_class=promptV_diff_class,
            schlink_experiments=schlink_experiments,
            sql_gen=sql_gen,
            diff_experiments=diff_experiments,
            all_experiments=all_experiments,
            experiment_settings=experiment_settings,
            **kwargs
        )
    else:
        # Call sequential_generate_sql to process data sequentially
        all_experiments, experiment_settings = sequential_generate_sql(
            data=data,
            data_path=data_path,
            exp_dir=experiment_dir,
            model=model,
            model_decomp=model_decomp,
            n_exps=n_exps,
            sql_gen_method=sql_gen_method,
            promptV_gen=promptV_gen,
            promptV_diff_class=promptV_diff_class,
            schlink_experiments=schlink_experiments,
            sql_gen=sql_gen,
            diff_experiments=diff_experiments,
            all_experiments=all_experiments,
            experiment_settings=experiment_settings,
            **kwargs
        )

    config["experiment_settings"] = experiment_settings

    # Save all experiments to a single JSON file
    with open(os.path.join(experiment_dir, exp_name + ".json"), "w") as f:
        json.dump(all_experiments, f, indent=4)
    logger.info(f"All experiments saved to {os.path.join(experiment_dir, exp_name + '.json')}")
    
    with open(os.path.join(experiment_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
                
    # Run evaluation if requested
    if eval_flag:
        logger.info("Running evaluation...")
        eval_results = evaluate_alerce_queries(
            data_path=data_path,
            data_format=data_format,
            model_name=model_name,
            save_path=save_path,
            exp_name=exp_name,
            **kwargs
        )
        logger.info(f"Evaluation completed and saved.")
    
        # Self-correction step
        if self_correction_flag:
            from main_selfcorrection import run_selfcorrection

            logger.info("Running self-correction step...")
            
            # Process and apply self-correction where needed
            corrected_experiments = run_selfcorrection(
                data_path=data_path,
                data_format=data_format,
                model_name=model_name,
                save_path=save_path,
                exp_name=exp_name,
                promptV_sc=promptV_sc,
                eval=eval_flag,  # Use the eval_flag to determine whether to evaluate after correction
                batch=batch,
                **kwargs,
            )
            
            logger.info(f"Self-correction completed and saved to {os.path.join(experiment_dir, f'corrected_{exp_name}.json')}")

            # Join evaluation results from the original and corrected experiments
            final_eval_results = join_eval_results(
                save_path=save_path,
                model_name=model_name,
                exp_name=exp_name,
                )

            # Get a summary of the evaluation stats
            corrected_stats = get_evaluation_stats(final_eval_results)
            
            # Also get the original stats for comparison
            original_stats = get_evaluation_stats(eval_results)
            
            # Print a comparison
            logger.info("=== Evaluation Results Comparison ===")
            logger.info(f"====== Gold queries with execution errors: {original_stats.get('gold_errors',0)} ===")
            logger.info(f"====== Predicted queries with execution errors: {original_stats.get('pred_errors',0)} === \n")
            logger.info(f"====== Gold queries with execution errors self-correction: {corrected_stats.get('gold_errors',0)} ===")
            logger.info(f"====== Predicted queries with execution errors self-correction: {corrected_stats.get('pred_errors',0)} === \n")
            logger.info(f"Original OID Success Rate: {original_stats.get('oids_success_rate', 0):.4f}")
            logger.info(f"Corrected OID Success Rate: {corrected_stats.get('oids_success_rate', 0):.4f}")
            logger.info(f"Original OID F1 Score: {original_stats.get('oids_f1', 0):.4f}")
            logger.info(f"Corrected OID F1 Score: {corrected_stats.get('oids_f1', 0):.4f}")
            logger.info(f"Original Error Rate: {original_stats.get('error_rate', 0):.4f}")
            logger.info(f"Corrected Error Rate: {corrected_stats.get('error_rate', 0):.4f}")

            logger.info("=== Evaluation Results by Difficulty ===")
            for difficulty in DifficultyLevel.get_valid_levels():
                logger.info(f"===== Evaluation Results for {difficulty} =====")
                logger.info(f"====== Gold queries with execution errors: {original_stats.get('by_difficulty').get(difficulty).get('gold_errors', 0)} ===")
                logger.info(f"====== Predicted queries with execution errors: {original_stats.get('by_difficulty').get(difficulty).get('pred_errors', 0)} === \n")
                logger.info(f"====== Gold queries with execution errors self-correction: {corrected_stats.get('by_difficulty').get(difficulty).get('gold_errors', 0)} ===")
                logger.info(f"====== Predicted queries with execution errors self-correction: {corrected_stats.get('by_difficulty').get(difficulty).get('pred_errors', 0)} === \n")
                
                logger.info(f"Original OID Success Rate for {difficulty}: {original_stats.get('by_difficulty').get(difficulty).get(f'oids_perfect_match_rate', 0):.4f}")
                logger.info(f"Corrected OID Success Rate for {difficulty}: {corrected_stats.get('by_difficulty').get(difficulty).get(f'oids_perfect_match_rate', 0):.4f}")
                logger.info(f"Original OID F1 Score for {difficulty}: {original_stats.get('by_difficulty').get(difficulty).get(f'columns_perfect_match_rate', 0):.4f}")
                logger.info(f"Corrected OID F1 Score for {difficulty}: {corrected_stats.get('by_difficulty').get(difficulty).get(f'columns_perfect_match_rate', 0):.4f}")
        else:
            # get a summary of the evaluation stats
            eval_stats = get_evaluation_stats(eval_results)
            
            logger.info("=== Evaluation Results ===")
            logger.info(f"Evaluation Results for {exp_name}:")
            logger.info(f"====== Gold queries with execution errors: {eval_stats.get('gold_errors',0)} ===")
            logger.info(f"====== Predicted queries with execution errors: {eval_stats.get('pred_errors',0)} === \n")
            logger.info(f"Oid Perfect Match Rate: {eval_stats.get('oids_perfect_match_rate', 0):.4f}")
            logger.info(f"Column Perfect Match Rate: {eval_stats.get('columns_perfect_match_rate', 0):.4f}")
            logger.info(f"Oid F1 Score: {eval_stats.get('oids_f1', 0):.4f}")
            logger.info(f"Column F1 Score: {eval_stats.get('columns_f1', 0):.4f}")
            logger.info(f"Error Rate: {eval_stats.get('error_rate', 0):.4f}")

            logger.info("=== Evaluation Results by Difficulty ===")
            for difficulty in DifficultyLevel.get_valid_levels():
                logger.info(f"===== Evaluation Results for {difficulty} =====")
                logger.info(f"====== Gold queries with execution errors: {eval_stats.get('by_difficulty').get(difficulty).get('gold_errors', 0)} ===")
                logger.info(f"====== Predicted queries with execution errors: {eval_stats.get('by_difficulty').get(difficulty).get('pred_errors', 0)} === \n")
                logger.info(f"Oid Perfect Match Rate for {difficulty}: {eval_stats.get('by_difficulty').get(difficulty).get(f'oids_perfect_match_rate', 0):.4f}")
                logger.info(f"Column Perfect Match Rate for {difficulty}: {eval_stats.get('by_difficulty').get(difficulty).get(f'columns_perfect_match_rate', 0):.4f}")
                
    else:
        logger.info("Evaluation not requested. Skipping evaluation step.")            

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM experiments.")
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="The path to the data file.")
    parser.add_argument("--data_format", type=str, default='csv', choices=["csv", "json"], help="The type of data to use (e.g., 'text', 'json', 'csv').")
    # Pipeline parameters
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.", )
    parser.add_argument("--exp_name", type=str, required=True, help="The name of the experiment to run.")
    parser.add_argument("--sql_gen_method", type=str, default="direct", choices=GenerationMethod.get_valid_methods(), help="The SQL generation method to use.")
    parser.add_argument("--n_exps", type=int, default=5, help="The number of experiments to run.")
    parser.add_argument("--save_path", type=str, default="./results", help="The path to save the results.")
    parser.add_argument("--eval", action='store_true', help="Whether to run evaluation after the experiment.")
    parser.add_argument("--self_correction", action='store_true', help="Whether to run self-correction after the experiment.")
    parser.add_argument("--schema_linking_method", type=bool, default=True, help="Whether to use schema linking method.")
    parser.add_argument("--promptV_sl", type=str, default="sl_v0", help="The version of the schema linking prompt to use.")
    parser.add_argument("--promptV_diff_class", type=str, default="diff_v0", help="The version of the difficulty classification prompt to use.")
    parser.add_argument("--promptV_gen", type=str, default="dir_v0", help="The version of the SQL generation prompt to use.")
    parser.add_argument("--promptV_sc", type=str, default="sc_v0", help="The version of the self-correction prompt to use.")
    # Evaluation parameters
    parser.add_argument("--num_processes", type=int, default=5, help="The number of parallel processes to use for evaluation.")
    parser.add_argument("--parallel", type=bool, default=True, help="Whether to use parallel processing for evaluation.")
    parser.add_argument("--access_time", type=int, default=2, help="The access time for the database.")
    parser.add_argument("--n_tries", type=int, default=5, help="The number of tries to run each query.")
    parser.add_argument("--alias_handling", type=bool, default=True, help="Whether to handle column name aliases.")
    # optional parameters
    parser.add_argument("--batch", action='store_true', help="Whether to make a batch request to the model.")
    parser.add_argument("--model_name_decomp", type=str, default=None, help="The name of the model to use for decomposition. If None, use the same model as model_name.")

    # model parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--max_new_tokens_decomp", type=int, default=1024, help="Max new tokens to generate for decomposition")
    parser.add_argument("--t", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top p")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run the experiment
    run_pipeline(
        data_path=args.data_path,
        data_format=args.data_format,
        model_name=args.model_name,
        save_path=args.save_path,
        exp_name=args.exp_name,
        n_exps=args.n_exps,
        eval=args.eval,
        self_correction=args.self_correction,
        batch=args.batch,
        schema_linking_method=args.schema_linking_method,
        sql_gen_method=args.sql_gen_method,
        promptV_sl = args.promptV_sl,
        promptV_diff_class = args.promptV_diff_class,
        promptV_gen = args.promptV_gen,
        promptV_sc = args.promptV_sc,
        num_processes=args.num_processes,
        parallel=args.parallel,
        access_time=args.access_time,
        n_tries=args.n_tries,
        alias_handling=args.alias_handling,
        model_name_decomp=args.model_name_decomp,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_decomp=args.max_new_tokens_decomp,
        t=args.t,
        top_p=args.top_p,
    )