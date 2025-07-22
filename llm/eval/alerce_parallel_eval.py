"""
Parallel processing utilities for ALeRCE database evaluation.
This module provides parallel processing capabilities for evaluating multiple SQL queries simultaneously.
"""
import os
import time
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
from eval.alerce_db_eval import compare_sql_queries, compare_sql_queries_multiple
from utils.utils import extract_sql
from utils.eval_utils import metrics_aggregation
from typing import Union, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query_in_parallel(args):
    """
    Process a single query comparison in a worker process.
    
    Args:
        args (tuple): A tuple containing:
            - req_id: The request ID
            - gold_query: The gold (reference) SQL query
            - pred_queries: The predicted SQL queries dictionary
            - access_time: Database connection timeout
            - n_tries: Number of attempts to run each query
            - alias_handling: Whether to handle column name aliases for identifiers
    
    Returns:
        List[dict]: A list of dictionaries containing the comparison results
    """
    req_id, gold_query, pred_queries, access_time, n_tries, alias_handling, metadata = args
    results = []
    
    # Take only valid predicted SQL queries
    if pred_queries is not None: pred_queries = {k: v for k, v in pred_queries.items() if v is not None} 
    # Skip if there are no valid predicted SQL queries
    if pred_queries is None or len(pred_queries) == 0:
        return results
    
    # If there is only one predicted SQL query
    if len(pred_queries) == 1:
        logger.info(f"Processing query with req_id: {req_id} (single prediction)")
        n_exp = next(iter(pred_queries.keys()))  # Get the only key
        pred_query = pred_queries[n_exp]
        
        # Extract SQL if it's wrapped in markdown code blocks
        pred_query = extract_sql(pred_query)
        
        # Compare the queries
        comparison = compare_sql_queries(
            sql_query_gold=gold_query,
            sql_query_pred=pred_query,
            access_time=access_time, 
            n_tries=n_tries,
            alias_handling=alias_handling
        )
        if comparison.get('error_gold') is not None:
            logger.info(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")
        
        # Add request ID and other metadata to result
        result = {
            "req_id": req_id,
            "n_exp": n_exp,
            "comparison": comparison
        }
        # Add metadata if present
        if metadata:
            for key, value in metadata.items():
                result[key] = value
                
        results.append(result)
        
    # If there are multiple predicted SQL queries
    else:
        logger.info(f"Processing {len(pred_queries)} queries with req_id: {req_id} (multiple predictions)")
        
        # Extract SQL if it's wrapped in markdown code blocks
        pred_queries = {k: extract_sql(v) for k, v in pred_queries.items()}
        
        comparison = compare_sql_queries_multiple(
            sql_query_gold=gold_query,
            sql_query_pred_dict=pred_queries,
            access_time=access_time,
            n_tries=n_tries,
            alias_handling=alias_handling
        )
        if comparison.get('error_gold') is not None:
            logger.info(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")

        # Add request ID and other metadata to result for each prediction
        for n_exp, comp_result in comparison.items():
            result = {
                "req_id": req_id,
                "n_exp": n_exp,
                "comparison": comp_result
            }
            # Add metadata if present
            if metadata:
                for key, value in metadata.items():
                    result[key] = value
                    
            results.append(result)
    
    return results

def error_handler(e):
    """
    Handler for errors in parallel execution.
    
    Args:
        e: The exception that was raised
    """
    print(f"Error in worker process: {e}", flush=True)
    print(f"Error type: {type(e)}", flush=True)
    if hasattr(e, '__cause__') and e.__cause__ is not None:
        print(f"Caused by: {e.__cause__}", flush=True)

def run_parallel_evaluation(
    database: pd.DataFrame,
    predicted_sqls: Dict[str, Dict[str, str]],
    access_time: int = 2,
    n_tries: int = 3,
    alias_handling: bool = True,
    num_processes: Union[int, None] = None,
):
    """
    Run SQL query evaluations in parallel using multiprocessing.
    
    Args:
        database: The database containing gold queries
        predicted_sqls: Dictionary of predicted SQL queries {req_id: {n_exp: sql_query}}
        access_time: Database connection timeout
        n_tries: Number of attempts to run each query
        alias_handling: Whether to handle column name aliases for identifiers
        num_processes: Number of processes to use (defaults to CPU count)
    
    Returns:
        dict: A dictionary containing the evaluation results
    """
    start_time = time.time()
    
    # Determine number of processes to use
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        
    # Ensure database has req_id and gold_query columns
    if 'req_id' not in database.columns or 'gold_query' not in database.columns:
        raise ValueError("Database must have 'req_id' and 'gold_query' columns")
    
    # Process the predicted SQL queries
    req_ids = database['req_id'].astype(str).tolist()
    gold_queries = database['gold_query'].tolist()
    
    # Prepare arguments for parallel processing
    args_list = []
    for req_id, gold_query in zip(req_ids, gold_queries):
        # Extract metadata if available
        metadata = {}
        if 'difficulty' in database.columns:
            metadata['difficulty'] = database.loc[database['req_id'].astype(str) == str(req_id), 'difficulty'].values[0]
        if 'table_info' in database.columns:
            metadata['gold_tables'] = database.loc[database['req_id'].astype(str) == str(req_id), 'table_info'].values[0]
        if 'type' in database.columns:
            metadata['type'] = database.loc[database['req_id'].astype(str) == str(req_id), 'type'].values[0]

        args_list.append((
            req_id,
            gold_query,
            predicted_sqls.get(req_id, None),
            access_time,
            n_tries,
            alias_handling,
            metadata
        ))
    
    # Run evaluations in parallel
    print(f"Running {len(args_list)} evaluations in parallel with {num_processes} processes")
    results = []
    
    # Avoid creating pool if no args to process
    if not args_list:
        print("No valid queries to evaluate.")
        return {
            'detailed_results': [],
            'aggregate_metrics': {},
            'metadata': {
                'total_queries': len(database),
                'evaluated_queries': 0,
                'skipped_queries': len(database),
                'evaluation_time': 0,
                'parallel_processes': num_processes
            }
        }
    
    with mp.Pool(processes=num_processes) as pool:
        # Map the function to the arguments
        async_results = [
            pool.apply_async(
                process_query_in_parallel, 
                args=(args,),
                error_callback=error_handler
            ) 
            for args in args_list
        ]
        
        # Wait for all processes to complete and collect results
        for async_result in async_results:
            try:
                result = async_result.get()
                if result:  # Make sure result is not None or empty
                    results.extend(result)
            except Exception as e:
                print(f"Failed to get result: {e}")
                
    # Close the pool
    # pool.close()
    # pool.join()
    
    # Calculate aggregate metrics (same as sequential version)
    aggregate_metrics = metrics_aggregation(results)
    
    # Create final result
    end_time = time.time()

    n_runs = len(predicted_sqls[database['req_id'].astype(str).tolist()[0]])
    evaluation_result = {
        'detailed_results': results,
        'aggregate_metrics': aggregate_metrics,
        'metadata': {
            'total_queries': len(predicted_sqls)*n_runs,
            'evaluated_queries': len(results),
            'skipped_queries': len(predicted_sqls)*n_runs - len(results),
            'evaluation_time': end_time - start_time,
            'parallel_processes': num_processes
        }
    }
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
    return evaluation_result
