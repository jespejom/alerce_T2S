

"""
Main script for self-correction of SQL queries.
This script loads the initial SQL predictions, applies self-correction,
and saves the corrected SQL queries.
"""

import os
import json
import logging
import argparse
import time
import pandas as pd
from typing import Dict, Any, Union

from model.llm import LLMModel
from utils.utils import load_dataset
from utils.llm_utils import load_sql_model
from utils.eval_utils import join_eval_results, get_evaluation_stats
from selfcorrection.self_correction import SelfCorrection
from evaluation import evaluate_alerce_queries
from prompts.DBSchemaPrompts import alerce_tables_desc, schema_all_cntxV2_indx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sequential_self_correction(
        data: pd.DataFrame,
        model: LLMModel,
        all_experiments: Dict[str, Any],
        eval_data: list,
        experiment_settings: Dict[str, Any],
        promptV_sc: str,
) -> tuple:
    """
    Sequential self-correction function.
    This function applies self-correction to SQL queries that had execution errors.
    
    Args:
        all_experiments (Dict[str, Any]): The original experiment results.
        eval_data (list): Evaluation data containing error information.
        self_correction_model (SelfCorrection): The self-correction model to use.
        experiment_dir (str): The directory where experiment data is stored.
        exp_name (str): The name of the experiment.
    
    Returns:
        tuple: A tuple containing:
            - Dict[str, Any]: The corrected experiment results.
            - Dict[str, Any]: Settings and metadata about the self-correction process.
    """
    logger.info("Starting sequential self-correction process...")
    start_time = time.time()
    corrected_experiments = all_experiments.copy()
    
    # Track metrics
    total_corrections = 0
    
    for eval in eval_data:
        req_id = eval.get('req_id')
        n_exp = eval.get('n_exp')
        if eval['comparison']['error_pred']:
            pred_tables = corrected_experiments[req_id][n_exp].get('pred_tables')
            # Initialize the self-correction method
            self_correction_model = SelfCorrection(
                model=model,
                tables_list=pred_tables,
                prompt_version=promptV_sc,
            )

            request = data.loc[data['req_id'] == int(req_id), 'request'].values[0]
            sql_query = corrected_experiments[req_id][n_exp].get('sql_query')
            error_msg = eval['comparison']['error_pred']

            # Apply self-correction to this query
            corrected_sql, correction_response = self_correction_model.correct_query(
                query=request,
                sql_query=sql_query,
                sql_error=error_msg,
                n=1
            )

            corrected_experiments[req_id][n_exp]['original_sql_query'] = corrected_experiments[req_id][n_exp]['sql_query']
            corrected_experiments[req_id][n_exp]['original_sql_response'] = corrected_experiments[req_id][n_exp]['sql_response']
            corrected_experiments[req_id][n_exp]['sql_query'] = corrected_sql[0]
            corrected_experiments[req_id][n_exp]['sql_response'] = correction_response
            corrected_experiments[req_id][n_exp]['correction_applied'] = True
            total_corrections += 1
        
        else:
            # If there's no specific error_pred, this query might not need correction
            corrected_experiments[req_id][n_exp]['original_sql_query'] = corrected_experiments[req_id][n_exp]['sql_query']
            corrected_experiments[req_id][n_exp]['original_sql_response'] = corrected_experiments[req_id][n_exp]['sql_response']
            corrected_experiments[req_id][n_exp]['sql_query'] = None
            corrected_experiments[req_id][n_exp]['sql_response'] = None
            corrected_experiments[req_id][n_exp]['correction_applied'] = False

    experiment_settings['self_correction'] = self_correction_model.to_dict()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
        
    logger.info(f"Sequential self-correction completed with {total_corrections} corrections in {elapsed_time:.2f}s")

    return corrected_experiments, experiment_settings

def batch_self_correction(
        data: pd.DataFrame,
        model: LLMModel,
        all_experiments: Dict[str, Any],
        eval_data: list,
        experiment_settings: Dict[str, Any],
        exp_name: str,
        experiment_dir: str,
        promptV_sc: str,
) -> tuple:
    """
    Batch self-correction function.
    This function applies self-correction to SQL queries that had execution errors in batches.
    
    Args:
        data (pd.DataFrame): The dataset containing the original queries.
        model (LLMModel): The language model to use for corrections.
        all_experiments (Dict[str, Any]): The original experiment results.
        eval_data (list): Evaluation data containing error information.
        experiment_settings (Dict[str, Any]): Settings and metadata about the experiments.
        exp_name (str, optional): The name of the experiment. Used for saving batch files.
        experiment_dir (str, optional): The directory to save batch files.

    Returns:
        tuple: A tuple containing:
            - Dict[str, Any]: The corrected experiment results.
            - Dict[str, Any]: Updated experiment settings with self-correction info.
    """
    logger.info("Starting batch self-correction process...")
    start_time = time.time()
    corrected_experiments = all_experiments.copy()
    
    # Track metrics
    total_corrections = 0
    
    # Prepare batch requests
    batch_messages = []    
    # Process each query that needs correction
    for eval in eval_data:
        req_id = eval.get('req_id')
        n_exp = eval.get('n_exp')

        if eval['comparison']['error_pred']:
            pred_tables = corrected_experiments[req_id][n_exp].get('pred_tables')
            
            # Initialize the self-correction method
            self_correction_model = SelfCorrection(
                model=model,
                tables_list=pred_tables,
                prompt_version=promptV_sc,
            )
            
            request = data.loc[data['req_id'] == int(req_id), 'request'].values[0]
            sql_query = corrected_experiments[req_id][n_exp].get('sql_query')
            error_msg = eval['comparison']['error_pred']
            
            logger.info(f"Adding request {req_id}, experiment {n_exp} to batch for correction")
            
            # Create a batch message for this query
            sc_messages = self_correction_model.return_batch(
                query=request,
                sql_query=sql_query,
                sql_error=error_msg
            )
            
            # Add to our batch with metadata
            batch_messages.append({"batch_id": f"{req_id}-{n_exp}", "messages": sc_messages, "n": 1})
            
            corrected_experiments[req_id][n_exp]['original_sql_query'] = corrected_experiments[req_id][n_exp]['sql_query']
            corrected_experiments[req_id][n_exp]['original_sql_response'] = corrected_experiments[req_id][n_exp]['sql_response']
            corrected_experiments[req_id][n_exp]['sql_query'] = None
            corrected_experiments[req_id][n_exp]['sql_response'] = None
            corrected_experiments[req_id][n_exp]['correction_applied'] = True
            total_corrections += 1
            
        else:
            # If there's no specific error_pred, this query might not need correction
            corrected_experiments[req_id][n_exp]['original_sql_query'] = corrected_experiments[req_id][n_exp]['sql_query']
            corrected_experiments[req_id][n_exp]['original_sql_response'] = corrected_experiments[req_id][n_exp]['sql_response']
            corrected_experiments[req_id][n_exp]['sql_query'] = None
            corrected_experiments[req_id][n_exp]['sql_response'] = None
            corrected_experiments[req_id][n_exp]['correction_applied'] = False

    experiment_settings['self_correction'] = self_correction_model.to_dict()
    
    # send the batch messages to the model
    model.run_batch(
        batch_messages,
        os.path.join(experiment_dir, "batch_sc_messages.jsonl"),
        exp_name
    )

    # wait for the model to finish
    # set a timeout for the batch response
    logger.info("Waiting for the model to finish processing the batch requests...")
    while True:
        batch_response = model.get_batch(
            os.path.join(experiment_dir, "batch_sc_messages_response.jsonl"),
        )
        time.sleep(300)  # wait for 5 minutes
        if batch_response is not None:
            break
        else:
            logger.info("Waiting for the model to finish...")

    for batch_key in batch_response.keys():
        # get the request id and experiment number
        # Only self-correction requests are in the batch
        req_id = batch_key.split("-")[0]
        n_exp = batch_key.split("-")[1]
        for null_exp in batch_response[batch_key]['responses'].keys(): # just one response
            # get the SQL query from the model response
            corrected_experiments[req_id][n_exp]["sql_query"] = batch_response[batch_key]['responses'][null_exp]
            corrected_experiments[req_id][n_exp]["sql_response"] = batch_response[batch_key]
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
        
    logger.info(f"Batch self-correction completed with {total_corrections} corrections in {elapsed_time:.2f}s")
    
    return corrected_experiments, experiment_settings
    
def run_selfcorrection(
    data_path: str,
    data_format: str,
    model_name: str,
    save_path: str,
    exp_name: str,
    promptV_sc: str,
    eval: bool = True,
    batch: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the self-correction process on an existing experiment.

    Args:
        data_path (str): Path to the original dataset.
        data_format (str): Format of the dataset (json, csv, etc.)
        model_name (str): The name/path of the model to use for self-correction.
        save_path (str): Base path where the original experiment results are stored.
        exp_name (str): Name of the original experiment.
        eval (bool): Whether to run evaluation after correction.
        batch (bool): Whether to use batch processing for corrections.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Any]: Dictionary with the corrected experiment results.
        
    Raises:
        FileNotFoundError: If required experiment files are not found.
        ValueError: If the experiment data format is invalid.
        Exception: For other unexpected errors during the correction process.
    """
    start_time = time.time()

    # Load the data
    data = load_dataset(data_path, data_format)

    # Load the model
    correction_model = load_sql_model(model_name, **kwargs)

    # Extract model directory name from full path if needed
    model_dir = os.path.basename(model_name)
    
    # Define paths for the experiment data
    experiment_dir = os.path.join(save_path, model_dir, exp_name)
    exp_file_path = os.path.join(experiment_dir, f"{exp_name}.json")
    eval_file_path = os.path.join(experiment_dir, f"eval_{exp_name}.json")
    
    # Check if the experiment files exist
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    if not os.path.exists(exp_file_path):
        raise FileNotFoundError(f"Experiment file not found: {exp_file_path}")
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file_path}")
    
    # Load the experiment data
    logger.info(f"Loading original experiment data from: {exp_file_path}")
    with open(exp_file_path, 'r') as f:
        all_experiments = json.load(f)
    
    # Load the evaluation data that contains error information
    logger.info(f"Loading evaluation data from: {eval_file_path}")
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    
    detailed_results = eval_data.get('detailed_results')
    if not detailed_results:
        raise ValueError(f"Evaluation data in {eval_file_path} is empty or invalid.")
    

    # Load configuration settings
    with open(os.path.join(experiment_dir, "config.json"), 'r') as f:
        config = json.load(f)
    experiment_settings = config.get('experiment_settings', {})
        
    # Apply self-correction
    logger.info("Applying self-correction to queries with errors")

    if batch:
        logger.info("Using batch mode for self-correction")
        corrected_experiments, sc_experiments_settings = batch_self_correction(
            data=data,
            model=correction_model,
            all_experiments=all_experiments,
            eval_data=detailed_results,
            experiment_settings=experiment_settings,
            experiment_dir=experiment_dir,
            exp_name=exp_name,
            promptV_sc=promptV_sc,
        )
    else:
        logger.info("Using sequential mode for self-correction")
        corrected_experiments, sc_experiments_settings = sequential_self_correction(
            data=data,
            model=correction_model,
            all_experiments=all_experiments,
            eval_data=detailed_results,
            experiment_settings=experiment_settings,
            promptV_sc=promptV_sc,
        )

    # Save the corrected results
    corrected_exp_name = f"corrected_{exp_name}"
    corrected_file_path = os.path.join(experiment_dir, f"{corrected_exp_name}.json")
    logger.info(f"Saving corrected experiments to: {corrected_file_path}")

    # Save the corrected experiments with the settings
    with open(corrected_file_path, 'w') as f:
        json.dump(corrected_experiments, f, indent=4)
    
    # Load and save in the configuration file the settings and prompts used
    with open(os.path.join(experiment_dir, "config.json"), 'r') as f:
        config = json.load(f)
    # Add the self-correction settings to the config
    config["experiment_settings"] = sc_experiments_settings
    # Keep the original experiment settings
    # Save the updated config
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
        
    # Run evaluation on the corrected results if requested
    if eval:
        logger.info("Running evaluation on corrected queries")
        
        # Evaluate the corrected queries
        corrected_eval_results = evaluate_alerce_queries(
            data_path=data_path,
            data_format=data_format,
            model_name=model_name,
            save_path=save_path,
            exp_name=exp_name,  # Use the experiments-only file
            # experiment_dir=experiment_dir,
            self_correction=True,
            **kwargs
        )
        
        # Join evaluation results from the original and corrected experiments
        final_eval_results = join_eval_results(
            save_path=save_path,
            model_name=model_name,
            exp_name=exp_name,
            )
        
        # Get a summary of the evaluation stats
        corrected_stats = get_evaluation_stats(final_eval_results)
        
        # Also get the original stats for comparison
        original_stats = get_evaluation_stats(eval_data)
        
        # Print a comparison
        logger.info("=== Evaluation Results Comparison ===")
        logger.info(f"====== Gold queries with execution errors: {original_stats.get('gold_errors',0)} ===")
        logger.info(f"====== Predicted queries with execution errors: {original_stats.get('pred_errors',0)} === \n")
        logger.info(f"====== Gold queries with execution errors self-correction: {corrected_stats.get('gold_errors',0)} ===")
        logger.info(f"====== Predicted queries with execution errors self-correction: {corrected_stats.get('pred_errors',0)} === \n")
        
        logger.info(f"Original Success Rate: {original_stats.get('oids_success_rate', 0):.4f}")
        logger.info(f"Corrected Success Rate: {corrected_stats.get('oids_success_rate', 0):.4f}")
        logger.info(f"Original F1 Score: {original_stats.get('oids_f1', 0):.4f}")
        logger.info(f"Corrected F1 Score: {corrected_stats.get('oids_f1', 0):.4f}")
        logger.info(f"Original Error Rate: {original_stats.get('error_rate', 0):.4f}")
        logger.info(f"Corrected Error Rate: {corrected_stats.get('error_rate', 0):.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Self-correction process completed in {total_time:.2f} seconds")
    
    return corrected_experiments

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SQL self-correction on existing experiment results.")
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--data_format", type=str, default="csv", help="Format of the dataset (json, csv, etc.)")
    # Pipeline parameters
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to use for self-correction.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the original experiment to correct.")
    parser.add_argument("--save_path", type=str, default="./results", help="Base path where results are saved.")
    parser.add_argument("--eval", action="store_true", help="Whether to run evaluation after correction.")
    parser.add_argument("--promptV_sc", type=str, default="sc_v0", help="The version of the self-correction prompt to use.")
    # Evaluation parameters
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes for parallel evaluation.")
    parser.add_argument("--parallel", action="store_true", help="Whether to use parallel processing for evaluation.")
    parser.add_argument("--access_time", type=int, default=2, help="The access time for the database.")
    parser.add_argument("--n_tries", type=int, default=5, help="The number of tries to run each query.")
    parser.add_argument("--alias_handling",  action="store_true", help="Whether to handle column name aliases.")
    # optional parameters
    parser.add_argument("--batch", action='store_true', help="Use batch mode for self-correction.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Run the self-correction process
    run_selfcorrection(
        data_path=args.data_path,
        data_format=args.data_format,
        model_name=args.model_name,
        save_path=args.save_path,
        exp_name=args.exp_name,
        eval=args.eval,
        batch=args.batch,  # Pass batch argument
        promptV_sc=args.promptV_sc,
        num_processes=args.num_processes,
        parallel=args.parallel,
        access_time=args.access_time,
        n_tries=args.n_tries,
        alias_handling=args.alias_handling,
    )
    logger.info("Self-correction process completed successfully.")