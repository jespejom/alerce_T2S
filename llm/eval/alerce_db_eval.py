import os
import time
import json
import pandas as pd
import numpy as np
import multiprocessing as mp
from utils.alerce_utils import run_sql_alerce
from utils.utils import extract_sql
from utils.eval_utils import get_query_columns, replace_renamed_columns, metrics_aggregation
from typing import Union, Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_results_oids(gold_result, pred_result, alias_handling=True, sql_query_pred=None):
    """
    Compare the identifiers results of two SQL queries and return the precision, recall, and F1 score.
    
    Args:
        gold_result (pandas.DataFrame): The expected identifiers result of the SQL query.
        pred_result (pandas.DataFrame): The predicted identifiers result of the SQL query.
        alias_handling (bool): Whether to handle column name aliases for identifiers
        sql_query_pred (str): The predicted SQL query. Used for extracting column info if alias_handling is True.
    Returns:
        dict: A dictionary containing the precision, recall, and F1 score of the comparison of identifiers.
    """
    
    # If the gold_result is empty, raise an error
    if gold_result is None or gold_result.empty:
        raise ValueError("The gold result is empty. Cannot compare with the predicted result.")
    
    # Check if the predicted results are empty
    if pred_result is None or pred_result.empty:
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_result) if gold_result is not None else 0,
            'size_pred': 0,
            'comparison_type': 'oids',
            'gold_id_col': None,
            'pred_id_col': None
        }
    
    # Working copies of dataframes
    gold_df = gold_result.copy()
    pred_df = pred_result.copy()

    # remove deduplication
    gold_df = gold_df.loc[:, ~gold_df.columns.duplicated()]
    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]

    # Define potential identifier columns and their aliases
    # identifier_columns = {
    #     'oid': ['oid', 'ztf_identifier', 'ztf identifier', 'ztf_oid', 'object', 'ztf', 
    #             'ztf_id', 'ztf object identifier', 'unique_object_identifier', 
    #             'object_identifier', 'ztf_object_id'],
    #     'oid_catalog': ['oid_catalog', 'catalog_id', 'catalog_identifier', 'catalog_oid'],
    #     'objectidps1': ['objectidps1', 'ps1_id', 'ps1_identifier', 'panstarrs_id'],
    #     'classifier_name': ['classifier_name', 'classifier', 'model_name'],
    #     'count': ['count', 'n_count', 'num_count', 'total_count']
    # }

    # define identifier columns as a list for priority search ordered by relevance
    identifier_columns = ["candid", "oid", "oid_catalog", "objectidps1", "classifier_name", "count"]
    
    # Find identifier column in each dataframe
    gold_id_col = None
    pred_id_col = None
    
    # First try to find exact match for primary identifier columns
    # for id_type, aliases in identifier_columns.items():
    for id_type in identifier_columns:
        if id_type in gold_df.columns:
            gold_id_col = id_type
            break
    for id_type in identifier_columns:
        if id_type in pred_df.columns:
            pred_id_col = id_type
            break
    
    # If no exact match, look for aliases if alias handling is enabled
    if alias_handling:
        # if gold_id_col is None:
        #     for id_type in identifier_columns:
        #         for alias in aliases:
        #             if alias in gold_df.columns:
        #                 gold_id_col = alias
        #                 break
        #         if gold_id_col:
        #             break
        
        if pred_id_col is None:
            # Get the formatted columns from the SQL query
            pred_cols_formatted = get_query_columns(sql_query_pred)
            pred_cols_name = [j['name'] for j in pred_cols_formatted] # get the names of the columns
            pred_cols_real = [j['real_name'] for j in pred_cols_formatted] # get the real names of the columns
            # Replace the renamed columns in the predicted columns
            pred_cols_final = replace_renamed_columns(pred_cols_name, pred_cols_real, pred_result.columns.tolist())
            pred_cols_final = [col.lower() for col in pred_cols_final]  # normalize to lowercase
            
            for i, id_type in enumerate(identifier_columns):
                if id_type in pred_cols_final:
                    pred_id_col = pred_result.columns.tolist()[pred_cols_final.index(id_type)]
                    break
    
    # If still no identifier column, try to find any column that might contain 'oid' in its name
    if gold_id_col is None:
        for col in gold_df.columns:
            if 'oid' in col.lower() or 'id' in col.lower():
                gold_id_col = col
                break
    
    if pred_id_col is None:
        for col in pred_df.columns:
            if 'oid' in col.lower() or 'id' in col.lower():
                pred_id_col = col
                break
    
    # If we still don't have identifier columns, return zeros
    if gold_id_col is None or pred_id_col is None:
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_df),
            'size_pred': len(pred_df),
            'comparison_type': 'no_identifiers',
            'gold_id_col': gold_id_col,
            'pred_id_col': pred_id_col
        }
    
    # Get the identifiers from each dataframe
    gold_identifiers = set(gold_df[gold_id_col].astype(str).values)
    pred_identifiers = set(pred_df[pred_id_col].astype(str).values)

    # Calculate precision, recall, and F1 score
    true_positives = len(gold_identifiers.intersection(pred_identifiers))
    false_positives = len(pred_identifiers - gold_identifiers)
    false_negatives = len(gold_identifiers - pred_identifiers)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Perfect match: both precision and recall must be 1
    perfect_match = 1 if precision == 1.0 and recall == 1.0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'perfect_match': perfect_match,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'size_gold': len(gold_df[gold_id_col]),
        'size_pred': len(pred_df[pred_id_col]),
        'comparison_type': 'oids',
        'gold_id_col': gold_id_col,
        'pred_id_col': pred_id_col
    }

def compare_results_columns(gold_result, pred_result):
    """
    Compare the columns results of two SQL queries and return the precision, recall, and F1 score.
    
    Args:
        gold_result (pandas.DataFrame): The expected columns result of the SQL query.
        pred_result (pandas.DataFrame): The predicted columns result of the SQL query.
    Returns:
        dict: A dictionary containing the precision, recall, and F1 score of the comparison of columns.
    """
    # if the gold_result is empty, raise an error
    if gold_result is None or gold_result.empty:
        raise ValueError("The gold result is empty. Cannot compare with the predicted result.")
    
    # Check if the predicted results is none
    if pred_result is None :
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'gold_columns': gold_result.columns.tolist(),
            'pred_columns': [],
            'size_gold': len(gold_result.columns) if gold_result is not None else 0,
            'size_pred': 0,
            'comparison_type': 'columns',
        }

    # Get the identifiers and columns of the tables
    gold_columns = set(gold_result.columns)
    pred_columns = set(pred_result.columns)

    # Calculate precision, recall, and F1 score
    true_positives = len(gold_columns.intersection(pred_columns))
    false_positives = len(pred_columns - gold_columns)
    false_negatives = len(gold_columns - pred_columns)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Perfect match for columns: recall must be 1 (all gold columns are present)
    perfect_match = 1 if recall == 1.0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'perfect_match': perfect_match,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'gold_columns': gold_result.columns.tolist(),
        'pred_columns': pred_result.columns.tolist(),   
        'size_gold': len(gold_result.columns),
        'size_pred': len(pred_result.columns),
        'comparison_type': 'columns',
    }


def compare_results_columns_formatted(sql_pred, gold_result, pred_result):
    """
    Compare the columns of two SQL queries and return the precision, recall, and F1 score.
    The function handles renamed columns and returns the final column names.
    Args:
        sql_pred (str): The predicted SQL query.
        gold_result (pandas.DataFrame): The expected columns result of the SQL query.
        pred_result (pandas.DataFrame): The predicted columns result of the SQL query.
    Returns:
        dict: A dictionary containing the precision, recall, and F1 score of the comparison of columns.
    """

    # If the gold_result is empty, raise an error
    if gold_result is None:
        raise ValueError("Gold result is empty")

    if pred_result is None:
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'gold_columns': gold_result.columns.tolist(),
            'pred_columns': [],
            'size_gold': len(gold_result.columns) if gold_result is not None else 0,
            'size_pred': 0,
            'comparison_type': 'columns',
        }
    # Get the identifiers and columns of the tables
    gold_columns = set(gold_result.columns)
    pred_columns = set(pred_result.columns)
    # Get the formatted columns from the SQL query
    pred_cols_formatted = get_query_columns(sql_pred)
    pred_cols_name = [j['name'] for j in pred_cols_formatted] # get the names of the columns
    pred_cols_real = [j['real_name'] for j in pred_cols_formatted] # get the real names of the columns
    # Replace the renamed columns in the predicted columns
    pred_cols_final = replace_renamed_columns(pred_cols_name, pred_cols_real, pred_result.columns.tolist())

    # Obtain the final set of predicted columns
    if pred_cols_final is not None:
        pred_columns_final = set(pred_cols_final)

        true_positives = len(gold_columns.intersection(pred_columns_final))
        false_positives = len(pred_columns_final - gold_columns)
        false_negatives = len(gold_columns - pred_columns_final)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Perfect match for columns: recall must be 1 (all gold columns are present)
        perfect_match = 1 if recall == 1.0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'perfect_match': perfect_match,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'gold_columns': gold_result.columns.tolist(),
            'pred_columns': pred_result.columns.tolist(),   
            'pred_columns_formatted': pred_cols_final,
            'size_gold': len(gold_result.columns),
            'size_pred': len(pred_cols_final),
            'comparison_type': 'columns',
        }
    
    else:
        true_positives = len(gold_columns.intersection(pred_columns))
        false_positives = len(pred_columns - gold_columns)
        false_negatives = len(gold_columns - pred_columns)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Perfect match for columns: recall must be 1 (all gold columns are present)
        perfect_match = 1 if recall == 1.0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'perfect_match': perfect_match,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'gold_columns': gold_result.columns.tolist(),
            'pred_columns': pred_result.columns.tolist(),   
            'pred_columns_formatted': [],
            'size_gold': len(gold_result.columns),
            'size_pred': len(pred_columns),
            'comparison_type': 'columns',
        }


def compare_sql_queries(
    sql_query_gold: str,
    sql_query_pred: str,
    access_time: int = 2,
    n_tries: int = 3,
    alias_handling: bool = True,
) -> dict:
    """
    Run two SQL queries in the ALeRCE database and compare their results.
    It compares the identifiers (oid, catalog_id, ...) and the columns of the tables.
    It returns the matched identifiers and columns using precision and recall.

    Args:
        sql_query_gold (str): The expected SQL query to be executed.
        sql_query_pred (str): The predicted SQL query to be executed.
        access_time (int): Database connection timeout (2 for default, 10 for extended)
        n_tries (int): Number of attempts to run each query
        alias_handling (bool): Whether to handle column name aliases for identifiers

    Returns:
        dict: A dictionary containing the precision, recall, and F1 score of the comparison.
    """
    # Execute the gold SQL query
    gold_result, gold_error, gold_execution_time = run_sql_alerce(sql_query_gold, access_time=access_time, n_tries=n_tries, query_time=True)

    # If the gold query fails, return 0 for all metrics
    if gold_error is not None:
        print(f"Error executing gold query: {gold_error}")
        null_result = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': 0,
            'size_pred': 0,
            'comparison_type': 'error',
        }
        return {"oids": null_result, "columns": null_result, "columns_formatted": null_result,
                "sql_gold": sql_query_gold, "sql_pred": sql_query_pred,
                "execution_time_gold": None, "execution_time_pred": None,
                "error_gold": str(gold_error), "error_pred": None}

    # Execute the predicted SQL query
    pred_result, pred_error, pred_execution_time = run_sql_alerce(sql_query_pred, access_time=access_time, n_tries=n_tries, query_time=True)
    
    # If the predicted query fails, return 0 for all metrics and include the error
    if pred_error is not None:
        null_result = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_result) if gold_result is not None else 0,
            'size_pred': 0,
            'comparison_type': 'error',
        }
        return {"oids": null_result, "columns": null_result, "columns_formatted": null_result,
                "sql_gold": sql_query_gold, "sql_pred": sql_query_pred,
                "execution_time_gold": gold_execution_time, "execution_time_pred": pred_execution_time,
                "error_gold": None, "error_pred": str(pred_error)}

    # Compare the results
    comparison_result = {}
    
    # Compare identifiers (OIDs)
    try:
        comparison_result['oids'] = compare_results_oids(gold_result, pred_result, alias_handling=alias_handling, sql_query_pred=sql_query_pred)
    except Exception as e:
        comparison_result['oids'] = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_result) if gold_result is not None else 0,
            'size_pred': len(pred_result) if pred_result is not None else 0,
            'comparison_type': 'error',
            'error': str(e)
        }
    
    # Compare columns
    try:
        comparison_result['columns'] = compare_results_columns(gold_result, pred_result)
    except Exception as e:
        comparison_result['columns'] = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_result.columns) if gold_result is not None else 0,
            'size_pred': len(pred_result.columns) if pred_result is not None else 0,
            'comparison_type': 'error',
            'error': str(e)
        }

    # Compare columns with formatted SQL query
    try:
        comparison_result['columns_formatted'] = compare_results_columns_formatted(
            sql_query_pred, gold_result, pred_result
        )
    except Exception as e:
        comparison_result['columns_formatted'] = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': len(gold_result.columns) if gold_result is not None else 0,
            'size_pred': len(pred_result.columns) if pred_result is not None else 0,
            'comparison_type': 'error',
            'error': str(e)
        }
    
    # Add the queries and execution times to the result
    comparison_result['sql_gold'] = sql_query_gold
    comparison_result['sql_pred'] = sql_query_pred
    comparison_result['execution_time_gold'] = gold_execution_time
    comparison_result['execution_time_pred'] = pred_execution_time
    comparison_result['error_gold'] = None
    comparison_result['error_pred'] = None
    
    return comparison_result


def compare_sql_queries_multiple(
    sql_query_gold: str,
    sql_query_pred_dict: dict,
    access_time: int = 2,
    n_tries: int = 5,
    alias_handling: bool = True,
) -> dict:
    """
    Compare multiple predicted SQL queries against a single gold SQL query.
    
    This function takes a single gold (reference) SQL query and a dictionary of
    predicted SQL queries, executes them all, and compares their results. It evaluates both
    the identifiers (oid, catalog_id, etc.) and columns, returning precision, recall, 
    and F1 scores for each prediction.
    
    The function optimizes execution by detecting duplicate queries and reusing results
    rather than re-executing identical queries multiple times.

    Args:
        sql_query_gold (str): The expected (gold) SQL query to be executed
        sql_query_pred_dict (dict): Dictionary mapping experiment IDs to predicted SQL queries
                              (e.g., {id1: "SQL query 1", id2: "SQL query 2", ...})
        access_time (int): Database connection timeout in seconds
        n_tries (int): Number of attempts to run each query before giving up
        alias_handling (bool): Whether to handle column name aliases for identifiers
                              (e.g., match 'oid' with 'ztf_id' or 'object_id')

    Returns:
        dict: Dictionary mapping the same IDs from sql_query_pred_dict to their evaluation results
              Format: {id1: {comparison metrics}, id2: {comparison metrics}, ...}
              Each result contains metrics for both identifiers and columns.
    """

    # Validate input
    if not sql_query_gold or not isinstance(sql_query_gold, str):
        raise ValueError("Gold SQL query must be a non-empty string")
    
    if not sql_query_pred_dict or not isinstance(sql_query_pred_dict, dict):
        raise ValueError("Predicted SQL queries must be provided as a non-empty dictionary")
    
    # Initialize the result dictionary
    comparison_result = {}
    
    # Execute the gold SQL query
    gold_result, gold_error, gold_execution_time = run_sql_alerce(sql_query_gold, access_time=access_time, n_tries=n_tries, query_time=True)

    # If the gold query fails, return 0 for all metrics
    if gold_error is not None:
        print(f"Error executing gold query: {gold_error}")
        null_result = {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'perfect_match': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'size_gold': 0,
            'size_pred': 0,
            'comparison_type': 'error',
        }
        
        # If the gold query fails, we need to return the error for each predicted query
        for i in sql_query_pred_dict.keys():
            comparison_result[i] = {"oids": null_result, "columns": null_result, "columns_formatted": null_result,
                                    "sql_gold": sql_query_gold, "sql_pred": sql_query_pred_dict[i],
                                    "execution_time_gold": None, "execution_time_pred": None,
                                    "error_gold": str(gold_error), "error_pred": None}
        return comparison_result

    # Execute each predicted SQL query and compare results
    total_queries = len(sql_query_pred_dict)
    # Dictionary to track duplicate queries
    duplicate_queries = {}
    # processed_queries_cache = {}
    
    for idx, (i, sql_query_pred) in enumerate(sql_query_pred_dict.items()):
        # Show progress
        if idx % 2 == 0 or idx == total_queries - 1:
            logger.info(f"Processing query {idx + 1} of {total_queries} ({(idx + 1) / total_queries * 100:.1f}%)")

        # Check for duplicates in the predicted SQL queries to avoid redundant execution and comparison
        # If the predicted query is already in the comparison result, skip it
        # This is a simple way to handle duplicates, but it may not be the most efficient for large datasets.  
        # Maybe use a hash table or set for better performance.
        # # Check if this exact predicted query string has been processed before
        # if sql_query_pred in processed_queries_cache:
        #     comparison_result[i] = processed_queries_cache[sql_query_pred]
        #     continue
        if sql_query_pred in duplicate_queries.values():
            # Find the index of the duplicate query
            duplicate_index = list(duplicate_queries.values()).index(sql_query_pred)
            # Find the corresponding key in the dictionary
            duplicate_key = list(duplicate_queries.keys())[duplicate_index]
            # Reuse the result from the duplicate query
            comparison_result[i] = comparison_result[duplicate_key]
            continue
        # If the predicted query is not in the duplicate queries, add it to the dictionary
        else:
            duplicate_queries[i] = sql_query_pred

        # Execute the predicted SQL query
        pred_result, pred_error, pred_execution_time = run_sql_alerce(sql_query_pred, access_time=access_time, n_tries=n_tries, query_time=True)
        
        # If the predicted query fails, return 0 for all metrics and include the error
        if pred_error is not None:
            null_result = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'perfect_match': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'size_gold': len(gold_result) if gold_result is not None else 0,
                'size_pred': 0,
                'comparison_type': 'error',
            }
            comparison_result[i] = {"oids": null_result, "columns": null_result, "columns_formatted": null_result,
                                    "sql_gold": sql_query_gold, "sql_pred": sql_query_pred,
                                    "execution_time_gold": gold_execution_time, "execution_time_pred": pred_execution_time,
                                    "error_gold": None, "error_pred": str(pred_error)}
            continue
        
        # Compare the results
        comparison_result[i] = {}
        # Compare identifiers (OIDs)
        try:
            comparison_result[i]['oids'] = compare_results_oids(gold_result, pred_result, alias_handling=alias_handling, sql_query_pred=sql_query_pred)
        except Exception as e:
            comparison_result[i]['oids'] = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'perfect_match': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'size_gold': len(gold_result) if gold_result is not None else 0,
                'size_pred': len(pred_result) if pred_result is not None else 0,
                'comparison_type': 'error',
                'error': str(e)
            }
        # Compare columns
        try:
            comparison_result[i]['columns'] = compare_results_columns(gold_result, pred_result)
        except Exception as e:
            comparison_result[i]['columns'] = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'perfect_match': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'size_gold': len(gold_result.columns) if gold_result is not None else 0,
                'size_pred': len(pred_result.columns) if pred_result is not None else 0,
                'comparison_type': 'error',
                'error': str(e)
            }

        # Compare columns with formatted SQL query
        try:
            comparison_result[i]['columns_formatted'] = compare_results_columns_formatted(
                sql_query_pred, gold_result, pred_result
            )
        except Exception as e:
            comparison_result[i]['columns_formatted'] = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'perfect_match': 0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'size_gold': len(gold_result.columns) if gold_result is not None else 0,
                'size_pred': len(pred_result.columns) if pred_result is not None else 0,
                'comparison_type': 'error',
                'error': str(e)
            }
        # Add the queries and execution times to the result
        comparison_result[i]['sql_gold'] = sql_query_gold
        comparison_result[i]['sql_pred'] = sql_query_pred
        comparison_result[i]['execution_time_gold'] = gold_execution_time
        comparison_result[i]['execution_time_pred'] = pred_execution_time
        comparison_result[i]['error_gold'] = None
        comparison_result[i]['error_pred'] = None

    return comparison_result

def alerce_usercases_evaluation(
        database: pd.DataFrame,
        predicted_sqls: Dict[str, Dict[str, str]],
        access_time: int = 2,
        n_tries: int = 3,
        alias_handling: bool = True
) -> dict:
    """
    Evaluate the predicted SQL queries against the expected SQL queries.
    It compares the identifiers (oid, catalog_id, ...) and the columns of the tables.
    It returns the matched identifiers and columns using precision and recall.

    Args:
        database: A pandas DataFrame containing the database with the following columns:
            - req_id: The request ID of the SQL query
            - gold_query: The expected SQL query to be executed
        predicted_sqls: A dictionary containing the predicted SQL queries for each request ID. It should be in the format:
            {req_id1: {n_exp1: "SQL query 1", n_exp2: "SQL query 2", ...},
             req_id2: {n_exp1: "SQL query 1", n_exp2: "SQL query 2", ...}, 
             ...}
            - req_id: The request ID of the SQL query
            - n_exp: The experiment number of the SQL query
            - sql_query: The predicted SQL query to be executed
        access_time (int): Database connection timeout (2 for default, 10 for extended)
        n_tries (int): Number of attempts to run each query
        alias_handling (bool): Whether to handle column name aliases for identifiers
        
    Returns:
        dict: A dictionary containing evaluation results including:
            - detailed_results: Per-query evaluation metrics
            - aggregate_metrics: Overall metrics across all queries
            - metadata: Information about the evaluation run
    """
    start_time = time.time()
     
    # Ensure database has req_id and gold_query columns
    if 'req_id' not in database.columns or 'gold_query' not in database.columns:
        raise ValueError("Database must have 'req_id' and 'gold_query' columns")
    
    # Process the predicted SQL queries
    req_ids = database['req_id'].astype(str).tolist()
    gold_queries = database['gold_query'].tolist()
        
    # Run evaluations for each request ID
    logger.info(f"Running evaluations for {len(req_ids)} request IDs...")
    results = []
    # results_run = []

    for i, req_id in enumerate(req_ids):
        gold_query = gold_queries[i]
        pred_queries = predicted_sqls.get(req_id, None)
        # Show progress
        if i % 2 == 0 or i == len(req_ids) - 1:
            logger.info(f"Processing query {i + 1} of {len(req_ids)} with req_id: {req_id} ({(i + 1) / len(req_ids) * 100:.1f}%)")
        # Take only valid predicted SQL queries
        if pred_queries is not None: pred_queries = {k: v for k, v in pred_queries.items() if v is not None}
        # If there are no predicted SQL queries
        # or all of them are None, skip this request ID
        if pred_queries is None or len(pred_queries) == 0:
            logger.warning(f"No predicted SQL queries {pred_queries} found for request ID {req_id}. Skipping...")
            continue
        # If there is only one predicted SQL query
        elif len(pred_queries) == 1:
            # key can be any running experiment, from 0 to n  
            # pred_query = pred_queries.get('0', None)
            pred_query = pred_queries[list(pred_queries.keys())[0]]
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
                print(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")
                logger.error(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")
            
            # Add request ID and other metadata to result
            result = {
                "req_id": req_id,
                "n_exp": "0",
                "comparison": comparison
            }
            if 'difficulty' in database.columns: result["difficulty"] = database.loc[database['req_id'].astype(str) == str(req_id), 'difficulty'].values[0]
            if 'table_info' in database.columns: result["gold_tables"] = database.loc[database['req_id'].astype(str) == str(req_id), 'table_info'].values[0]
            if 'type' in database.columns: result["type"] = database.loc[database['req_id'].astype(str) == str(req_id), 'type'].values[0]
            results.append(result)
            
        # If there are multiple predicted SQL queries
        elif len(pred_queries) > 1:
            # If there are multiple predicted SQL queries, run a parallel evaluation
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
                print(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")
                logger.error(f"Execution error for gold query with ID {req_id}: {comparison['error_gold']}")

            # Add request ID and other metadata to result
            for n_exp, result in comparison.items():
                result = {
                    "req_id": req_id,
                    "n_exp": n_exp,
                    "comparison": result
                }
                if 'difficulty' in database.columns: result["difficulty"] = database.loc[database['req_id'].astype(str) == str(req_id), 'difficulty'].values[0]
                if 'table_info' in database.columns: result["gold_tables"] = database.loc[database['req_id'].astype(str) == str(req_id), 'table_info'].values[0]
                if 'type' in database.columns: result["type"] = database.loc[database['req_id'].astype(str) == str(req_id), 'type'].values[0]
                results.append(result)
            
        else:
            raise ValueError(f"Unexpected number of predicted SQL queries for request ID {req_id}: {len(pred_queries)}")
    
    # Aggregate metrics
    aggregate_metrics = metrics_aggregation(results=results)

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
            'evaluation_time': end_time - start_time
        }
    }
    
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds.")
    return evaluation_result


