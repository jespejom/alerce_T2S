"""
Main interface for ALeRCE database evaluation.
This module provides a unified interface for evaluating SQL queries
using either sequential or parallel processing.
"""
import os
import time
import json
import pandas as pd
import numpy as np
from utils.utils import load_dataset
from utils.eval_utils import join_eval_results, get_evaluation_stats
from eval.alerce_db_eval import alerce_usercases_evaluation
from eval.alerce_parallel_eval import run_parallel_evaluation
from constants import GenerationMethod, DifficultyLevel
from typing import Union
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_alerce_queries(
        data_path: str,
        data_format: str,
        model_name: str,
        save_path: str,
        exp_name: str,
        parallel: bool = True,
        num_processes: Union[int, None] = None,
        access_time: int = 2,
        n_tries: int = 5,
        alias_handling: bool = True,
        experiment_dir: Union[str, None] = None,
        self_correction: bool = False,
        **kwargs
    ):
    """
    Evaluate predicted SQL queries against gold queries using either
    sequential or parallel processing.
    
    Args:
        data_path (str): Path to the gold queries database
        data_format (str): Format of the gold queries database (e.g., "json", "csv")
        model_name (str): Name of the model to use for evaluation
        save_path (str): Path to save the evaluation results
        exp_name (str): Name of the experiment to evaluate
        parallel (bool): Whether to use parallel processing
        num_processes (int): Number of processes for parallel execution (None=auto)
        access_time (int): Database connection timeout
        n_tries (int): Number of attempts to run each query
        alias_handling (bool): Whether to handle column name aliases
    
    Returns:
        dict: A dictionary containing the evaluation results
    """

    # load the database
    database = load_dataset(data_path, data_format)

    model_dir = os.path.basename(model_name)  # Extract just the model name from potential paths


    # load the predicted SQLs
    if experiment_dir is not None:
        # If experiment_dir is provided, use it directly
        pass
    else:
        # Otherwise, construct the path using model_name and exp_name
        experiment_dir = os.path.join(save_path, model_dir, exp_name)
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory {experiment_dir} does not exist.")
    
    if self_correction: 
        logger.info("Using self-corrected SQLs for evaluation.")
        exp_name = f"corrected_{exp_name}"
    
    with open(os.path.join(experiment_dir, exp_name + ".json"), "r") as f:
        all_experiments = json.load(f)
    logger.info(f"Loaded predicted SQLs from {os.path.join(experiment_dir, exp_name+'.json')}")

    # predicted_sqls = get_predicted_sqls(all_experiments)
    # Validate that all_experiments is a dictionary
    if not isinstance(all_experiments, dict):
        raise ValueError("Expected 'all_experiments' to be a dictionary, but got type: {}".format(type(all_experiments)))

    # Get the ids of the gold queries
    predicted_sqls = {}
    gold_id = database.req_id.astype(str).tolist()

    for id in gold_id:
        if id in all_experiments:
            pred_sqls_reqid = {}
            for n_exp, exp_data in all_experiments[id].items():
                if 'sql_query' in exp_data:
                    pred_sqls_reqid[n_exp] = exp_data.get('sql_query', None)
                else:
                    logger.warning(f"Missing 'sql_query' or invalid run number '{n_exp}' for ID {id}.")
            predicted_sqls[id] = pred_sqls_reqid
        else:
            logger.warning(f"ID {id} not found in predicted SQLs.")        

    start_time = time.time()
    # Choose evaluation method based on parallel flag
    if parallel:
        print(f"Using parallel evaluation with {'auto-detected' if num_processes is None else num_processes} processes")
        results = run_parallel_evaluation(
            database=database,
            predicted_sqls=predicted_sqls,
            access_time=access_time,
            n_tries=n_tries,
            alias_handling=alias_handling,
            num_processes=num_processes
        )
    else:
        print("Using sequential evaluation")
        results = alerce_usercases_evaluation(
            database=database,
            predicted_sqls=predicted_sqls,
            access_time=access_time,
            n_tries=n_tries,
            alias_handling=alias_handling
        )
    
    # Create final result
    results['metadata']['access_time'] = access_time
    results['metadata']['n_tries'] = n_tries
    results['metadata']['alias_handling'] = alias_handling
    results['metadata']['execution_method'] = 'parallel' if parallel else 'sequential'
    results['metadata']['total_execution_time'] = time.time() - start_time
    
    # Save the evaluation results
    with open(os.path.join(experiment_dir, f'eval_{exp_name}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluation results saved to {os.path.join(experiment_dir, f'eval_{exp_name}.json')}")
    logger.info("Experiment completed successfully!")

    return results
    
def parse_args():
    """
    Parse command line arguments for the evaluation script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SQL queries using ALeRCE database.")
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True, help="The path to the data file.")
    parser.add_argument("--data_format", type=str, default='csv', choices=["csv", "json"], help="The type of data to use (e.g., 'text', 'json', 'csv').")
    # Pipeline parameters
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model to use.", )
    parser.add_argument("--exp_name", type=str, required=True, help="The name of the experiment to run.")
    parser.add_argument("--save_path", type=str, default="./results", help="The path to save the results.")
    # Evaluation parameters
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes for parallel evaluation")
    parser.add_argument('--parallel', action='store_true', help="Use parallel evaluation")
    parser.add_argument('--access_time', type=int, default=2, help="Database connection timeout")
    parser.add_argument('--n_tries', type=int, default=5, help="Number of attempts to run each query")
    parser.add_argument('--alias_handling', action='store_true', help="Handle column name aliases")
    #  Optional parameters
    # choice between self-correction, no self-correction and both
    parser.add_argument('--self_correction', choices=['none', 'self-correction', 'both'], default='none', help="Whether to use self-correction, no self-correction or both.")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # n parallel processes
    n_processes = args.num_processes if args.parallel else None
    # evaluate the experiment
    if args.self_correction == 'self-correction':
        logger.info("Running evaluation with self-correction ...")
        eval_results = evaluate_alerce_queries( 
                data_path=args.data_path,
                data_format=args.data_format,
                model_name=args.model_name,
                save_path=args.save_path,
                exp_name=args.exp_name,
                parallel=args.parallel,
                access_time=args.access_time,
                n_tries=args.n_tries,
                alias_handling=args.alias_handling,
                num_processes=n_processes,
                self_correction=True
            )

        # Join evaluation results from the original and corrected experiments
        eval_results = join_eval_results(
            save_path=args.save_path,
            model_name=args.model_name,
            exp_name=args.exp_name,
        )
    elif args.self_correction == 'both':
        logger.info("Running evaluation without self-correction...")
        eval_results = evaluate_alerce_queries( 
                data_path=args.data_path,
                data_format=args.data_format,
                model_name=args.model_name,
                save_path=args.save_path,
                exp_name=args.exp_name,
                parallel=args.parallel,
                access_time=args.access_time,
                n_tries=args.n_tries,
                alias_handling=args.alias_handling,
                num_processes=n_processes,
                self_correction=False
            )
        # get a summary of the evaluation stats
        eval_stats = get_evaluation_stats(eval_results)
        logger.info("=== Evaluation Results ===")
        logger.info(f"Evaluation Results for {args.exp_name}:")
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
        
        logger.info("Running evaluation with self-correction...")
        eval_results = evaluate_alerce_queries( 
                data_path=args.data_path,
                data_format=args.data_format,
                model_name=args.model_name,
                save_path=args.save_path,
                exp_name=args.exp_name,
                parallel=args.parallel,
                access_time=args.access_time,
                n_tries=args.n_tries,
                alias_handling=args.alias_handling,
                num_processes=n_processes,
                self_correction=True
            )
        # Join evaluation results from the original and corrected experiments
        eval_results = join_eval_results(
            save_path=args.save_path,
            model_name=args.model_name,
            exp_name=args.exp_name,
            )

    else:
        logger.info("Running evaluation without self-correction...")
        eval_results = evaluate_alerce_queries(
            data_path=args.data_path,
            data_format=args.data_format,
            model_name=args.model_name,
            save_path=args.save_path,
            exp_name=args.exp_name,
            parallel=args.parallel,
            access_time=args.access_time,
            n_tries=args.n_tries,
            alias_handling=args.alias_handling,
            num_processes=n_processes,
            self_correction=False
        )

    # get a summary of the evaluation stats
    eval_stats = get_evaluation_stats(eval_results)
    logger.info("=== Evaluation Results ===")
    logger.info(f"Evaluation Results for {args.exp_name}:")
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

