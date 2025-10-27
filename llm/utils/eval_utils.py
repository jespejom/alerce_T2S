import os
import json
from typing import List, Dict
import numpy as np

def metrics_aggregation(results: List[dict], take_error_gold: bool = False) -> dict:
    """
    Aggregate the metrics from multiple SQL query comparisons.

    Args:
        results (list): A list of dictionaries containing the comparison results for each SQL query.
        take_error_gold (bool): Flag indicating if it is considered queries where the gold query failed. True means they are considered.

    Returns:
        dict: A dictionary containing the aggregated metrics for OIDs and columns, 
              including means and standard deviations.
    """

    # Initialize overall metrics
    oids_metrics = {
        'precision_values': [], 'recall_values': [], 'f1_values': [],
        'perfect_match_count': [], 
        'count': 0  # Number of successful (non-error) OID comparisons
    }
    
    columns_metrics = {
        'precision_values': [], 'recall_values': [], 'f1_values': [],
        'perfect_match_count': [],
        'count': 0  # Number of successful (non-error) column comparisons
    }

    columns_formatted = {
        'precision_values': [], 'recall_values': [], 'f1_values': [],
        'perfect_match_count': [],
        'count': 0  # Number of successful (non-error) column comparisons
    }
    
    error_counts = {
        'gold': 0,
        'pred': 0,
        'gold_list': [],
    }
    
    execution_times = {
        'gold_values': [], 
        'pred_values': []
        # Counts for these will be len(list)
    }

    # Initialize metrics for by_n_run, by_req_id, and by_difficulty
    metrics_by_n_run = {}
    metrics_by_req_id = {}
    metrics_by_difficulty = {}
    # New data structures for metrics by difficulty and runs
    metrics_by_difficulty_runs = {}
    metrics_by_difficulty_req_id = {}
    
    # Calculate aggregate metrics
    for result in results:
        comparison = result['comparison']
        n_run_val = str(result['n_exp']) # Ensure n_exp is string for dict key
        req_id_val = str(result['req_id']) # Ensure req_id is string for dict key
        difficulty_val = result['difficulty']

        # Initialize group-specific metrics if not present
        if n_run_val not in metrics_by_n_run:
            metrics_by_n_run[n_run_val] = {
                'oids_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_fmt_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'error_counts': {'gold': 0, 'pred': 0, 'gold_list': []},
                'execution_times': {'gold_values': [], 'pred_values': []},
                'group_total_items': 0
            }
        if req_id_val not in metrics_by_req_id:
            metrics_by_req_id[req_id_val] = {
                'oids_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_fmt_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'error_counts': {'gold': 0, 'pred': 0, 'gold_list': []},
                'execution_times': {'gold_values': [], 'pred_values': []},
                'group_total_items': 0
            }
        if difficulty_val not in metrics_by_difficulty:
            metrics_by_difficulty[difficulty_val] = {
                'oids_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_fmt_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'error_counts': {'gold': 0, 'pred': 0, 'gold_list': []},
                'execution_times': {'gold_values': [], 'pred_values': []},
                'group_total_items': 0
            }
        
        # Initialize metrics by difficulty and runs
        if difficulty_val not in metrics_by_difficulty_runs:
            metrics_by_difficulty_runs[difficulty_val] = {}
        if n_run_val not in metrics_by_difficulty_runs[difficulty_val]:
            metrics_by_difficulty_runs[difficulty_val][n_run_val] = {
                'oids_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_fmt_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'error_counts': {'gold': 0, 'pred': 0, 'gold_list': []},
                'execution_times': {'gold_values': [], 'pred_values': []},
                'group_total_items': 0
            }
            
        # Initialize metrics by difficulty and request ID
        if difficulty_val not in metrics_by_difficulty_req_id:
            metrics_by_difficulty_req_id[difficulty_val] = {}
        if req_id_val not in metrics_by_difficulty_req_id[difficulty_val]:
            metrics_by_difficulty_req_id[difficulty_val][req_id_val] = {
                'oids_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'columns_fmt_metrics': {'precision_values': [], 'recall_values': [], 'f1_values': [], 'perfect_match_count': [], 'count': 0},
                'error_counts': {'gold': 0, 'pred': 0, 'gold_list': []},
                'execution_times': {'gold_values': [], 'pred_values': []},
                'group_total_items': 0
            }
            
        metrics_by_n_run[n_run_val]['group_total_items'] += 1
        metrics_by_req_id[req_id_val]['group_total_items'] += 1
        metrics_by_difficulty[difficulty_val]['group_total_items'] += 1
        metrics_by_difficulty_runs[difficulty_val][n_run_val]['group_total_items'] += 1
        metrics_by_difficulty_req_id[difficulty_val][req_id_val]['group_total_items'] += 1
        
        # Count errors (overall and group-specific)
        if comparison.get('error_gold'):
            error_counts['gold'] += 1
            error_counts['gold_list'].append(result['req_id'])
            metrics_by_n_run[n_run_val]['error_counts']['gold'] += 1
            metrics_by_req_id[req_id_val]['error_counts']['gold'] += 1
            metrics_by_difficulty[difficulty_val]['error_counts']['gold'] += 1
            metrics_by_difficulty[difficulty_val]['error_counts']['gold_list'].append(result['req_id'])
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['error_counts']['gold'] += 1
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['error_counts']['gold_list'].append(result['req_id'])
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['error_counts']['gold'] += 1
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['error_counts']['gold_list'].append(result['req_id'])
            if not take_error_gold:
                continue # Skip further processing for this result if gold query failed
            
        if comparison.get('error_pred'):
            error_counts['pred'] += 1
            metrics_by_n_run[n_run_val]['error_counts']['pred'] += 1
            metrics_by_req_id[req_id_val]['error_counts']['pred'] += 1
            metrics_by_difficulty[difficulty_val]['error_counts']['pred'] += 1
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['error_counts']['pred'] += 1
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['error_counts']['pred'] += 1
        
        # Aggregate OIDs metrics (overall and group-specific)
        if 'oids' in comparison:
            oid_result = comparison['oids']
        
            oids_metrics['count'] += 1
            oids_metrics['precision_values'].append(oid_result.get('precision', 0))
            oids_metrics['recall_values'].append(oid_result.get('recall', 0))
            oids_metrics['f1_values'].append(oid_result.get('f1_score', 0))
            oids_metrics['perfect_match_count'].append(oid_result.get('perfect_match', 0))

            # Group-specific OID metrics
            group_oids_n_run = metrics_by_n_run[n_run_val]['oids_metrics']
            group_oids_n_run['count'] += 1
            group_oids_n_run['precision_values'].append(oid_result.get('precision', 0))
            group_oids_n_run['recall_values'].append(oid_result.get('recall', 0))
            group_oids_n_run['f1_values'].append(oid_result.get('f1_score', 0))
            group_oids_n_run['perfect_match_count'].append(oid_result.get('perfect_match', 0))
            metrics_by_n_run[n_run_val]['oids_metrics'] = group_oids_n_run

            # Group-specific OID metrics by req_id
            group_oids_req_id = metrics_by_req_id[req_id_val]['oids_metrics']
            group_oids_req_id['count'] += 1
            group_oids_req_id['precision_values'].append(oid_result.get('precision', 0))
            group_oids_req_id['recall_values'].append(oid_result.get('recall', 0))
            group_oids_req_id['f1_values'].append(oid_result.get('f1_score', 0))
            group_oids_req_id['perfect_match_count'].append(oid_result.get('perfect_match', 0))
            metrics_by_req_id[req_id_val]['oids_metrics'] = group_oids_req_id

            # Group-specific OID metrics by difficulty
            group_oids_difficulty = metrics_by_difficulty[difficulty_val]['oids_metrics']
            group_oids_difficulty['count'] += 1
            group_oids_difficulty['precision_values'].append(oid_result.get('precision', 0))
            group_oids_difficulty['recall_values'].append(oid_result.get('recall', 0))
            group_oids_difficulty['f1_values'].append(oid_result.get('f1_score', 0))
            group_oids_difficulty['perfect_match_count'].append(oid_result.get('perfect_match', 0))
            metrics_by_difficulty[difficulty_val]['oids_metrics'] = group_oids_difficulty
            
            # Group-specific OID metrics by difficulty and run
            group_oids_diff_run = metrics_by_difficulty_runs[difficulty_val][n_run_val]['oids_metrics']
            group_oids_diff_run['count'] += 1
            group_oids_diff_run['precision_values'].append(oid_result.get('precision', 0))
            group_oids_diff_run['recall_values'].append(oid_result.get('recall', 0))
            group_oids_diff_run['f1_values'].append(oid_result.get('f1_score', 0))
            group_oids_diff_run['perfect_match_count'].append(oid_result.get('perfect_match', 0))
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['oids_metrics'] = group_oids_diff_run
            
            # Group-specific OID metrics by difficulty and request ID
            group_oids_diff_req = metrics_by_difficulty_req_id[difficulty_val][req_id_val]['oids_metrics']
            group_oids_diff_req['count'] += 1
            group_oids_diff_req['precision_values'].append(oid_result.get('precision', 0))
            group_oids_diff_req['recall_values'].append(oid_result.get('recall', 0))
            group_oids_diff_req['f1_values'].append(oid_result.get('f1_score', 0))
            group_oids_diff_req['perfect_match_count'].append(oid_result.get('perfect_match', 0))
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['oids_metrics'] = group_oids_diff_req
        else:
            raise ValueError("Comparison result does not contain 'oids' key.")
                
        # Aggregate column metrics (overall and group-specific)
        if 'columns' in comparison:
            col_result = comparison['columns']
        
            columns_metrics['count'] += 1
            columns_metrics['precision_values'].append(col_result.get('precision', 0))
            columns_metrics['recall_values'].append(col_result.get('recall', 0))
            columns_metrics['f1_values'].append(col_result.get('f1_score', 0))
            columns_metrics['perfect_match_count'].append(col_result.get('perfect_match', 0))

            # Group-specific column metrics
            group_cols_n_run = metrics_by_n_run[n_run_val]['columns_metrics']
            group_cols_n_run['count'] += 1
            group_cols_n_run['precision_values'].append(col_result.get('precision', 0))
            group_cols_n_run['recall_values'].append(col_result.get('recall', 0))
            group_cols_n_run['f1_values'].append(col_result.get('f1_score', 0))
            group_cols_n_run['perfect_match_count'].append(col_result.get('perfect_match', 0))
            metrics_by_n_run[n_run_val]['columns_metrics'] = group_cols_n_run

            # Group-specific column metrics by req_id
            group_cols_req_id = metrics_by_req_id[req_id_val]['columns_metrics']
            group_cols_req_id['count'] += 1
            group_cols_req_id['precision_values'].append(col_result.get('precision', 0))
            group_cols_req_id['recall_values'].append(col_result.get('recall', 0))
            group_cols_req_id['f1_values'].append(col_result.get('f1_score', 0))
            group_cols_req_id['perfect_match_count'].append(col_result.get('perfect_match', 0))
            metrics_by_req_id[req_id_val]['columns_metrics'] = group_cols_req_id
            
            # Group-specific column metrics by difficulty
            group_cols_difficulty = metrics_by_difficulty[difficulty_val]['columns_metrics']
            group_cols_difficulty['count'] += 1
            group_cols_difficulty['precision_values'].append(col_result.get('precision', 0))
            group_cols_difficulty['recall_values'].append(col_result.get('recall', 0))
            group_cols_difficulty['f1_values'].append(col_result.get('f1_score', 0))
            group_cols_difficulty['perfect_match_count'].append(col_result.get('perfect_match', 0))
            metrics_by_difficulty[difficulty_val]['columns_metrics'] = group_cols_difficulty
            
            # Group-specific column metrics by difficulty and run
            group_cols_diff_run = metrics_by_difficulty_runs[difficulty_val][n_run_val]['columns_metrics']
            group_cols_diff_run['count'] += 1
            group_cols_diff_run['precision_values'].append(col_result.get('precision', 0))
            group_cols_diff_run['recall_values'].append(col_result.get('recall', 0))
            group_cols_diff_run['f1_values'].append(col_result.get('f1_score', 0))
            group_cols_diff_run['perfect_match_count'].append(col_result.get('perfect_match', 0))
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['columns_metrics'] = group_cols_diff_run
            
            # Group-specific column metrics by difficulty and request ID
            group_cols_diff_req = metrics_by_difficulty_req_id[difficulty_val][req_id_val]['columns_metrics']
            group_cols_diff_req['count'] += 1
            group_cols_diff_req['precision_values'].append(col_result.get('precision', 0))
            group_cols_diff_req['recall_values'].append(col_result.get('recall', 0))
            group_cols_diff_req['f1_values'].append(col_result.get('f1_score', 0))
            group_cols_diff_req['perfect_match_count'].append(col_result.get('perfect_match', 0))
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['columns_metrics'] = group_cols_diff_req
        else:
            raise ValueError("Comparison result does not contain 'columns' key.")
        
        if 'columns_formatted' in comparison:
            col_formatted_result = comparison['columns_formatted']
        
            columns_formatted['count'] += 1
            columns_formatted['precision_values'].append(col_formatted_result.get('precision', 0))
            columns_formatted['recall_values'].append(col_formatted_result.get('recall', 0))
            columns_formatted['f1_values'].append(col_formatted_result.get('f1_score', 0))
            columns_formatted['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))

            # Group-specific column formatted metrics
            group_cols_fmt_n_run = metrics_by_n_run[n_run_val]['columns_fmt_metrics']
            group_cols_fmt_n_run['count'] += 1
            group_cols_fmt_n_run['precision_values'].append(col_formatted_result.get('precision', 0))
            group_cols_fmt_n_run['recall_values'].append(col_formatted_result.get('recall', 0))
            group_cols_fmt_n_run['f1_values'].append(col_formatted_result.get('f1_score', 0))
            group_cols_fmt_n_run['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))
            metrics_by_n_run[n_run_val]['columns_fmt_metrics'] = group_cols_fmt_n_run

            # Group-specific column formatted metrics by req_id
            group_cols_fmt_req_id = metrics_by_req_id[req_id_val]['columns_fmt_metrics']
            group_cols_fmt_req_id['count'] += 1
            group_cols_fmt_req_id['precision_values'].append(col_formatted_result.get('precision', 0))
            group_cols_fmt_req_id['recall_values'].append(col_formatted_result.get('recall', 0))
            group_cols_fmt_req_id['f1_values'].append(col_formatted_result.get('f1_score', 0))
            group_cols_fmt_req_id['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))
            metrics_by_req_id[req_id_val]['columns_fmt_metrics'] = group_cols_fmt_req_id
            
            # Group-specific column formatted metrics by difficulty
            group_cols_fmt_difficulty = metrics_by_difficulty[difficulty_val]['columns_fmt_metrics']
            group_cols_fmt_difficulty['count'] += 1
            group_cols_fmt_difficulty['precision_values'].append(col_formatted_result.get('precision', 0))
            group_cols_fmt_difficulty['recall_values'].append(col_formatted_result.get('recall', 0))
            group_cols_fmt_difficulty['f1_values'].append(col_formatted_result.get('f1_score', 0))
            group_cols_fmt_difficulty['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))
            metrics_by_difficulty[difficulty_val]['columns_fmt_metrics'] = group_cols_fmt_difficulty
            
            # Group-specific column formatted metrics by difficulty and run
            group_cols_fmt_diff_run = metrics_by_difficulty_runs[difficulty_val][n_run_val]['columns_fmt_metrics']
            group_cols_fmt_diff_run['count'] += 1
            group_cols_fmt_diff_run['precision_values'].append(col_formatted_result.get('precision', 0))
            group_cols_fmt_diff_run['recall_values'].append(col_formatted_result.get('recall', 0))
            group_cols_fmt_diff_run['f1_values'].append(col_formatted_result.get('f1_score', 0))
            group_cols_fmt_diff_run['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['columns_fmt_metrics'] = group_cols_fmt_diff_run
            
            # Group-specific column formatted metrics by difficulty and request ID
            group_cols_fmt_diff_req = metrics_by_difficulty_req_id[difficulty_val][req_id_val]['columns_fmt_metrics']
            group_cols_fmt_diff_req['count'] += 1
            group_cols_fmt_diff_req['precision_values'].append(col_formatted_result.get('precision', 0))
            group_cols_fmt_diff_req['recall_values'].append(col_formatted_result.get('recall', 0))
            group_cols_fmt_diff_req['f1_values'].append(col_formatted_result.get('f1_score', 0))
            group_cols_fmt_diff_req['perfect_match_count'].append(col_formatted_result.get('perfect_match', 0))
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['columns_fmt_metrics'] = group_cols_fmt_diff_req

        # Aggregate execution times (overall and group-specific)
        time_gold = comparison.get('execution_time_gold')
        time_pred = comparison.get('execution_time_pred')

        if time_gold is not None:
            execution_times['gold_values'].append(time_gold)
            metrics_by_n_run[n_run_val]['execution_times']['gold_values'].append(time_gold)
            metrics_by_req_id[req_id_val]['execution_times']['gold_values'].append(time_gold)
            metrics_by_difficulty[difficulty_val]['execution_times']['gold_values'].append(time_gold)
            metrics_by_difficulty_runs[difficulty_val][n_run_val]['execution_times']['gold_values'].append(time_gold)
            metrics_by_difficulty_req_id[difficulty_val][req_id_val]['execution_times']['gold_values'].append(time_gold)
            
            if time_pred is not None: # Pred query also executed successfully
                execution_times['pred_values'].append(time_pred)
                metrics_by_n_run[n_run_val]['execution_times']['pred_values'].append(time_pred)
                metrics_by_req_id[req_id_val]['execution_times']['pred_values'].append(time_pred)
                metrics_by_difficulty[difficulty_val]['execution_times']['pred_values'].append(time_pred)
                metrics_by_difficulty_runs[difficulty_val][n_run_val]['execution_times']['pred_values'].append(time_pred)
                metrics_by_difficulty_req_id[difficulty_val][req_id_val]['execution_times']['pred_values'].append(time_pred)

    # Calculate averages and standard deviations for overall metrics
    aggregate_metrics = {
        'oids': {
            'precision': np.mean(oids_metrics['precision_values']).item() if oids_metrics['precision_values'] else 0.0,
            'precision_std': np.std(oids_metrics['precision_values'], ddof=1).item() if len(oids_metrics['precision_values']) > 1 else 0.0,
            'recall': np.mean(oids_metrics['recall_values']).item() if oids_metrics['recall_values'] else 0.0,
            'recall_std': np.std(oids_metrics['recall_values'], ddof=1).item() if len(oids_metrics['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(oids_metrics['f1_values']).item() if oids_metrics['f1_values'] else 0.0,
            'f1_score_std': np.std(oids_metrics['f1_values'], ddof=1).item() if len(oids_metrics['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(oids_metrics['perfect_match_count']).item() if oids_metrics['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(oids_metrics['perfect_match_count'], ddof=1).item() if len(oids_metrics['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(oids_metrics['perfect_match_count']).item() if oids_metrics['perfect_match_count'] else 0,
            'success_count': oids_metrics['count'],
            'total_count': len(results) # Grand total of items processed
        },
        'columns': {
            'precision': np.mean(columns_metrics['precision_values']).item() if columns_metrics['precision_values'] else 0.0,
            'precision_std': np.std(columns_metrics['precision_values'], ddof=1).item() if len(columns_metrics['precision_values']) > 1 else 0.0,
            'recall': np.mean(columns_metrics['recall_values']).item() if columns_metrics['recall_values'] else 0.0,
            'recall_std': np.std(columns_metrics['recall_values'], ddof=1).item() if len(columns_metrics['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(columns_metrics['f1_values']).item() if columns_metrics['f1_values'] else 0.0,
            'f1_score_std': np.std(columns_metrics['f1_values'], ddof=1).item() if len(columns_metrics['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(columns_metrics['perfect_match_count']).item() if columns_metrics['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(columns_metrics['perfect_match_count'], ddof=1).item() if len(columns_metrics['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(columns_metrics['perfect_match_count']).item() if columns_metrics['perfect_match_count'] else 0,
            'success_count': columns_metrics['count'],
            'total_count': len(results) # Grand total of items processed
        },
        'columns_formatted': {
            'precision': np.mean(columns_formatted['precision_values']).item() if columns_formatted['precision_values'] else 0.0,
            'precision_std': np.std(columns_formatted['precision_values'], ddof=1).item() if len(columns_formatted['precision_values']) > 1 else 0.0,
            'recall': np.mean(columns_formatted['recall_values']).item() if columns_formatted['recall_values'] else 0.0,
            'recall_std': np.std(columns_formatted['recall_values'], ddof=1).item() if len(columns_formatted['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(columns_formatted['f1_values']).item() if columns_formatted['f1_values'] else 0.0,
            'f1_score_std': np.std(columns_formatted['f1_values'], ddof=1).item() if len(columns_formatted['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(columns_formatted['perfect_match_count']).item() if columns_formatted['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(columns_formatted['perfect_match_count'], ddof=1).item() if len(columns_formatted['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(columns_formatted['perfect_match_count']).item() if columns_formatted['perfect_match_count'] else 0,
            'success_count': columns_formatted['count'],
            'total_count': len(results) # Grand total of items processed
        },
        'execution_times': {
            'avg_gold': np.mean(execution_times['gold_values']).item() if execution_times['gold_values'] else 0.0,
            'avg_gold_std': np.std(execution_times['gold_values'], ddof=1).item() if len(execution_times['gold_values']) > 1 else 0.0,
            'avg_pred': np.mean(execution_times['pred_values']).item() if execution_times['pred_values'] else 0.0,
            'avg_pred_std': np.std(execution_times['pred_values'], ddof=1).item() if len(execution_times['pred_values']) > 1 else 0.0,
            'count_gold_executions': len(execution_times['gold_values']),
            'count_pred_executions': len(execution_times['pred_values'])
        },
        'errors': {
            'gold_errors': error_counts['gold'],
            'gold_list': error_counts['gold_list'],
            'pred_errors': error_counts['pred'],
            'total_processed_after_gold_errors': len(results) - error_counts['gold']
        }
    }
    
    def _process_group_metrics_dict(group_metric_sums_container, group_total_items_val):
        processed_group = {}
        
        o_met_vals = group_metric_sums_container['oids_metrics']
        o_count = o_met_vals['count']
        processed_group['oids'] = {
            'precision': np.mean(o_met_vals['precision_values']).item() if o_met_vals['precision_values'] else 0.0,
            'precision_std': np.std(o_met_vals['precision_values'], ddof=1).item() if len(o_met_vals['precision_values']) > 1 else 0.0,
            'recall': np.mean(o_met_vals['recall_values']).item() if o_met_vals['recall_values'] else 0.0,
            'recall_std': np.std(o_met_vals['recall_values'], ddof=1).item() if len(o_met_vals['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(o_met_vals['f1_values']).item() if o_met_vals['f1_values'] else 0.0,
            'f1_score_std': np.std(o_met_vals['f1_values'], ddof=1).item() if len(o_met_vals['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(o_met_vals['perfect_match_count']).item() if o_met_vals['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(o_met_vals['perfect_match_count'], ddof=1).item() if len(o_met_vals['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(o_met_vals['perfect_match_count']).item() if o_met_vals['perfect_match_count'] else 0,
            'success_count': o_count,
            'group_total_items': group_total_items_val # Total items in this specific group
        }
        
        c_met_vals = group_metric_sums_container['columns_metrics']
        c_count = c_met_vals['count']
        processed_group['columns'] = {
            'precision': np.mean(c_met_vals['precision_values']).item() if c_met_vals['precision_values'] else 0.0,
            'precision_std': np.std(c_met_vals['precision_values'], ddof=1).item() if len(c_met_vals['precision_values']) > 1 else 0.0,
            'recall': np.mean(c_met_vals['recall_values']).item() if c_met_vals['recall_values'] else 0.0,
            'recall_std': np.std(c_met_vals['recall_values'], ddof=1).item() if len(c_met_vals['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(c_met_vals['f1_values']).item() if c_met_vals['f1_values'] else 0.0,
            'f1_score_std': np.std(c_met_vals['f1_values'], ddof=1).item() if len(c_met_vals['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(c_met_vals['perfect_match_count']).item() if c_met_vals['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(c_met_vals['perfect_match_count'], ddof=1).item() if len(c_met_vals['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(c_met_vals['perfect_match_count']).item() if c_met_vals['perfect_match_count'] else 0,
            'success_count': c_count,
            'group_total_items': group_total_items_val
        }

        cf_met_vals = group_metric_sums_container['columns_fmt_metrics']
        cf_count = cf_met_vals['count']
        processed_group['columns_formatted'] = {
            'precision': np.mean(cf_met_vals['precision_values']).item() if cf_met_vals['precision_values'] else 0.0,
            'precision_std': np.std(cf_met_vals['precision_values'], ddof=1).item() if len(cf_met_vals['precision_values']) > 1 else 0.0,
            'recall': np.mean(cf_met_vals['recall_values']).item() if cf_met_vals['recall_values'] else 0.0,
            'recall_std': np.std(cf_met_vals['recall_values'], ddof=1).item() if len(cf_met_vals['recall_values']) > 1 else 0.0,
            'f1_score': np.mean(cf_met_vals['f1_values']).item() if cf_met_vals['f1_values'] else 0.0,
            'f1_score_std': np.std(cf_met_vals['f1_values'], ddof=1).item() if len(cf_met_vals['f1_values']) > 1 else 0.0,
            'perfect_match_rate': np.mean(cf_met_vals['perfect_match_count']).item() if cf_met_vals['perfect_match_count'] else 0.0,
            'perfect_match_rate_std': np.std(cf_met_vals['perfect_match_count'], ddof=1).item() if len(cf_met_vals['perfect_match_count']) > 1 else 0.0,
            'perfect_match_count': np.sum(cf_met_vals['perfect_match_count']).item() if cf_met_vals['perfect_match_count'] else 0,
            'success_count': cf_count,
            'group_total_items': group_total_items_val
        }
        
        e_time_vals = group_metric_sums_container['execution_times']
        processed_group['execution_times'] = {
            'avg_gold': np.mean(e_time_vals['gold_values']).item() if e_time_vals['gold_values'] else 0.0,
            'avg_gold_std': np.std(e_time_vals['gold_values'], ddof=1).item() if len(e_time_vals['gold_values']) > 1 else 0.0,
            'avg_pred': np.mean(e_time_vals['pred_values']).item() if e_time_vals['pred_values'] else 0.0,
            'avg_pred_std': np.std(e_time_vals['pred_values'], ddof=1).item() if len(e_time_vals['pred_values']) > 1 else 0.0,
            'count_gold_executions': len(e_time_vals['gold_values']),
            'count_pred_executions': len(e_time_vals['pred_values'])
        }
        
        processed_group['errors'] = group_metric_sums_container['error_counts']
        processed_group['group_total_items'] = group_total_items_val # Redundant but matches prior structure for this key
        
        return processed_group

    final_metrics_by_n_run = {}
    for n_run_key, group_sums in metrics_by_n_run.items():
        if group_sums['oids_metrics']['perfect_match_count'] == [] and not take_error_gold: continue # skip queries where gold query failed
        final_metrics_by_n_run[n_run_key] = _process_group_metrics_dict(
            group_sums, 
            group_sums['group_total_items']
        )
    aggregate_metrics['by_n_run'] = final_metrics_by_n_run

    final_metrics_by_req_id = {}
    for req_id_key, group_sums in metrics_by_req_id.items():
        if group_sums['oids_metrics']['perfect_match_count'] == [] and not take_error_gold: continue # skip queries where gold query failed
        final_metrics_by_req_id[req_id_key] = _process_group_metrics_dict(
            group_sums, 
            group_sums['group_total_items']
        )
    aggregate_metrics['by_req_id'] = final_metrics_by_req_id

    final_metrics_by_difficulty = {}
    for diff_key, group_sums in metrics_by_difficulty.items():
        if group_sums['oids_metrics']['perfect_match_count'] == [] and not take_error_gold: continue # skip queries where gold query failed
        final_metrics_by_difficulty[diff_key] = _process_group_metrics_dict(
            group_sums, 
            group_sums['group_total_items'],
        )
    aggregate_metrics['by_difficulty'] = final_metrics_by_difficulty
    
    # After the metrics calculation for each group, now add the new by_difficulty_runs section
    # First, we'll process the metrics grouped by difficulty and runs
    final_metrics_by_difficulty_runs = {}
    for diff_key, runs_dict in metrics_by_difficulty_runs.items():
        # For each difficulty, calculate metrics per run
        runs_metrics = {}
        for run_key, run_metrics in runs_dict.items():
            if run_metrics['oids_metrics']['perfect_match_count'] == [] and not take_error_gold: continue # skip queries where gold query failed
            runs_metrics[run_key] = _process_group_metrics_dict(
                run_metrics,
                run_metrics['group_total_items'],
            )
        
        # Now calculate aggregate statistics across runs for this difficulty
        # First, extract metrics across all runs for this difficulty
        oids_precision_by_run = [metrics['oids']['precision'] for run, metrics in runs_metrics.items()]
        oids_recall_by_run = [metrics['oids']['recall'] for run, metrics in runs_metrics.items()]
        oids_f1_by_run = [metrics['oids']['f1_score'] for run, metrics in runs_metrics.items()]
        oids_pm_rate_by_run = [metrics['oids']['perfect_match_rate'] for run, metrics in runs_metrics.items()]
        
        cols_precision_by_run = [metrics['columns']['precision'] for run, metrics in runs_metrics.items()]
        cols_recall_by_run = [metrics['columns']['recall'] for run, metrics in runs_metrics.items()]
        cols_f1_by_run = [metrics['columns']['f1_score'] for run, metrics in runs_metrics.items()]
        cols_pm_rate_by_run = [metrics['columns']['perfect_match_rate'] for run, metrics in runs_metrics.items()]
        
        cols_fmt_precision_by_run = [metrics['columns_formatted']['precision'] for run, metrics in runs_metrics.items()]
        cols_fmt_recall_by_run = [metrics['columns_formatted']['recall'] for run, metrics in runs_metrics.items()]
        cols_fmt_f1_by_run = [metrics['columns_formatted']['f1_score'] for run, metrics in runs_metrics.items()]
        cols_fmt_pm_rate_by_run = [metrics['columns_formatted']['perfect_match_rate'] for run, metrics in runs_metrics.items()]
        
        # Calculate standard deviations across runs
        run_statistics = {
            'oids': {
                'precision': np.mean(oids_precision_by_run).item() if oids_precision_by_run else 0.0,
                'precision_std': np.std(oids_precision_by_run, ddof=1).item() if len(oids_precision_by_run) > 1 else 0.0,
                'recall': np.mean(oids_recall_by_run).item() if oids_recall_by_run else 0.0,
                'recall_std': np.std(oids_recall_by_run, ddof=1).item() if len(oids_recall_by_run) > 1 else 0.0,
                'f1_score': np.mean(oids_f1_by_run).item() if oids_f1_by_run else 0.0,
                'f1_score_std': np.std(oids_f1_by_run, ddof=1).item() if len(oids_f1_by_run) > 1 else 0.0,
                'perfect_match_rate': np.mean(oids_pm_rate_by_run).item() if oids_pm_rate_by_run else 0.0,
                'perfect_match_rate_std': np.std(oids_pm_rate_by_run, ddof=1).item() if len(oids_pm_rate_by_run) > 1 else 0.0
            },
            'columns': {
                'precision': np.mean(cols_precision_by_run).item() if cols_precision_by_run else 0.0,
                'precision_std': np.std(cols_precision_by_run, ddof=1).item() if len(cols_precision_by_run) > 1 else 0.0,
                'recall': np.mean(cols_recall_by_run).item() if cols_recall_by_run else 0.0,
                'recall_std': np.std(cols_recall_by_run, ddof=1).item() if len(cols_recall_by_run) > 1 else 0.0,
                'f1_score': np.mean(cols_f1_by_run).item() if cols_f1_by_run else 0.0,
                'f1_score_std': np.std(cols_f1_by_run, ddof=1).item() if len(cols_f1_by_run) > 1 else 0.0,
                'perfect_match_rate': np.mean(cols_pm_rate_by_run).item() if cols_pm_rate_by_run else 0.0,
                'perfect_match_rate_std': np.std(cols_pm_rate_by_run, ddof=1).item() if len(cols_pm_rate_by_run) > 1 else 0.0
            },
            'columns_formatted': {
                'precision': np.mean(cols_fmt_precision_by_run).item() if cols_fmt_precision_by_run else 0.0,
                'precision_std': np.std(cols_fmt_precision_by_run, ddof=1).item() if len(cols_fmt_precision_by_run) > 1 else 0.0,
                'recall': np.mean(cols_fmt_recall_by_run).item() if cols_fmt_recall_by_run else 0.0,
                'recall_std': np.std(cols_fmt_recall_by_run, ddof=1).item() if len(cols_fmt_recall_by_run) > 1 else 0.0,
                'f1_score': np.mean(cols_fmt_f1_by_run).item() if cols_fmt_f1_by_run else 0.0,
                'f1_score_std': np.std(cols_fmt_f1_by_run, ddof=1).item() if len(cols_fmt_f1_by_run) > 1 else 0.0,
                'perfect_match_rate': np.mean(cols_fmt_pm_rate_by_run).item() if cols_fmt_pm_rate_by_run else 0.0,
                'perfect_match_rate_std': np.std(cols_fmt_pm_rate_by_run, ddof=1).item() if len(cols_fmt_pm_rate_by_run) > 1 else 0.0
            },
            'runs_metrics': runs_metrics
        }
        
        final_metrics_by_difficulty_runs[diff_key] = run_statistics
    
    # Process metrics by difficulty and request ID
    final_metrics_by_difficulty_req_id = {}
    n_exps = len(metrics_by_n_run)
    for diff_key, req_ids_dict in metrics_by_difficulty_req_id.items():
        # For each difficulty, calculate metrics per request ID
        req_metrics = {}
        for req_id, req_id_metrics in req_ids_dict.items():
            if req_id_metrics['oids_metrics']['perfect_match_count'] == []: continue # skip queries where gold query failed
            elif len(req_id_metrics['oids_metrics']['perfect_match_count']) != n_exps:
                print(f"Warning: Request ID {req_id} has {req_id_metrics['oids_metrics']['perfect_match_count']} perfect matches, expected {n_exps}.")

            req_metrics[req_id] = _process_group_metrics_dict(
                req_id_metrics,
                req_id_metrics['group_total_items'],
            )
        
        # Extract metrics across all request IDs for this difficulty
        oids_precision_by_req = [metrics['oids']['precision'] for req, metrics in req_metrics.items()]
        oids_recall_by_req = [metrics['oids']['recall'] for req, metrics in req_metrics.items()]
        oids_f1_by_req = [metrics['oids']['f1_score'] for req, metrics in req_metrics.items()]
        oids_pm_rate_by_req = [metrics['oids']['perfect_match_rate'] for req, metrics in req_metrics.items()]
        
        cols_precision_by_req = [metrics['columns']['precision'] for req, metrics in req_metrics.items()]
        cols_recall_by_req = [metrics['columns']['recall'] for req, metrics in req_metrics.items()]
        cols_f1_by_req = [metrics['columns']['f1_score'] for req, metrics in req_metrics.items()]
        cols_pm_rate_by_req = [metrics['columns']['perfect_match_rate'] for req, metrics in req_metrics.items()]
        
        cols_fmt_precision_by_req = [metrics['columns_formatted']['precision'] for req, metrics in req_metrics.items()]
        cols_fmt_recall_by_req = [metrics['columns_formatted']['recall'] for req, metrics in req_metrics.items()]
        cols_fmt_f1_by_req = [metrics['columns_formatted']['f1_score'] for req, metrics in req_metrics.items()]
        cols_fmt_pm_rate_by_req = [metrics['columns_formatted']['perfect_match_rate'] for req, metrics in req_metrics.items()]
        
        # Calculate standard deviations across request IDs
        req_id_statistics = {
            'oids': {
                'precision': np.mean(oids_precision_by_req).item() if oids_precision_by_req else 0.0,
                'precision_std': np.std(oids_precision_by_req, ddof=1).item() if len(oids_precision_by_req) > 1 else 0.0,
                'recall': np.mean(oids_recall_by_req).item() if oids_recall_by_req else 0.0,
                'recall_std': np.std(oids_recall_by_req, ddof=1).item() if len(oids_recall_by_req) > 1 else 0.0,
                'f1_score': np.mean(oids_f1_by_req).item() if oids_f1_by_req else 0.0,
                'f1_score_std': np.std(oids_f1_by_req, ddof=1).item() if len(oids_f1_by_req) > 1 else 0.0,
                'perfect_match_rate': np.mean(oids_pm_rate_by_req).item() if oids_pm_rate_by_req else 0.0,
                'perfect_match_rate_std': np.std(oids_pm_rate_by_req, ddof=1).item() if len(oids_pm_rate_by_req) > 1 else 0.0
            },
            'columns': {
                'precision': np.mean(cols_precision_by_req).item() if cols_precision_by_req else 0.0,
                'precision_std': np.std(cols_precision_by_req, ddof=1).item() if len(cols_precision_by_req) > 1 else 0.0,
                'recall': np.mean(cols_recall_by_req).item() if cols_recall_by_req else 0.0,
                'recall_std': np.std(cols_recall_by_req, ddof=1).item() if len(cols_recall_by_req) > 1 else 0.0,
                'f1_score': np.mean(cols_f1_by_req).item() if cols_f1_by_req else 0.0,
                'f1_score_std': np.std(cols_f1_by_req, ddof=1).item() if len(cols_f1_by_req) > 1 else 0.0,
                'perfect_match_rate': np.mean(cols_pm_rate_by_req).item() if cols_pm_rate_by_req else 0.0,
                'perfect_match_rate_std': np.std(cols_pm_rate_by_req, ddof=1).item() if len(cols_pm_rate_by_req) > 1 else 0.0
            },
            'columns_formatted': {
                'precision': np.mean(cols_fmt_precision_by_req).item() if cols_fmt_precision_by_req else 0.0,
                'precision_std': np.std(cols_fmt_precision_by_req, ddof=1).item() if len(cols_fmt_precision_by_req) > 1 else 0.0,
                'recall': np.mean(cols_fmt_recall_by_req).item() if cols_fmt_recall_by_req else 0.0,
                'recall_std': np.std(cols_fmt_recall_by_req, ddof=1).item() if len(cols_fmt_recall_by_req) > 1 else 0.0,
                'f1_score': np.mean(cols_fmt_f1_by_req).item() if cols_fmt_f1_by_req else 0.0,
                'f1_score_std': np.std(cols_fmt_f1_by_req, ddof=1).item() if len(cols_fmt_f1_by_req) > 1 else 0.0,
                'perfect_match_rate': np.mean(cols_fmt_pm_rate_by_req).item() if cols_fmt_pm_rate_by_req else 0.0,
                'perfect_match_rate_std': np.std(cols_fmt_pm_rate_by_req, ddof=1).item() if len(cols_fmt_pm_rate_by_req) > 1 else 0.0
            },
            'req_id_metrics': req_metrics
        }
        
        final_metrics_by_difficulty_req_id[diff_key] = req_id_statistics
    
    # Add to the aggregate metrics
    aggregate_metrics['by_difficulty_runs'] = final_metrics_by_difficulty_runs
    aggregate_metrics['by_difficulty_req_id'] = final_metrics_by_difficulty_req_id
    
    return aggregate_metrics


def join_eval_results(
        save_path: str,
        model_name: str,
        exp_name: str
) -> dict:
    """
    Join the evaluation results from different runs.
    
    Args:
        save_path (str): The path to the directory where the results are saved.
        model_name (str): The name of the model used for evaluation.
        exp_name (str): The name of the experiment.
    
    Returns:
        dict: A dictionary containing the joined evaluation results.
    """
    
    # Extract model directory name from full path if needed
    model_dir = os.path.basename(model_name)
    
    # Define paths for the experiment data
    experiment_dir = os.path.join(save_path, model_dir, exp_name)
    exp_file_path = os.path.join(experiment_dir, f"{exp_name}.json")
    corrected_exp_file_path = os.path.join(experiment_dir, f"corrected_{exp_name}.json")
    eval_file_path = os.path.join(experiment_dir, f"eval_{exp_name}.json")
    corrected_eval_file_path = os.path.join(experiment_dir, f"eval_corrected_{exp_name}.json")
    
    # load experiment data
    with open(exp_file_path, 'r') as f:
        experiments = json.load(f)
    with open(corrected_exp_file_path, 'r') as f:
        corrected_experiments = json.load(f)

    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    eval_data_results = eval_data.get('detailed_results')

    with open(corrected_eval_file_path, 'r') as f:
        corrected_eval_data = json.load(f)
    corrected_eval_data_results = corrected_eval_data.get('detailed_results')
    # Create a dictionary for corrected evaluation data
    # to facilitate lookup by req_id and n_exp
    corrected_eval_data_dict = {}
    for corr_eval_exp in corrected_eval_data_results:
        req_id = str(corr_eval_exp['req_id'])
        n_exp = str(corr_eval_exp['n_exp'])
        if req_id not in corrected_eval_data_dict: corrected_eval_data_dict[req_id] = {}
        corrected_eval_data_dict[req_id][n_exp] = corr_eval_exp
    
    # Create a list to store the final evaluation data
    final_eval_data = []
    for eval_exp in eval_data_results:
        req_id = str(eval_exp['req_id'])
        n_exp = str(eval_exp['n_exp'])

        try: corrected_experiments[req_id][n_exp]['correction_applied']
        except:
            raise ValueError(
                f"Experiment {req_id} with n_exp {n_exp} does not have a correction applied. "
                "Please ensure the corrected experiments are properly structured."
            )
        
        if corrected_experiments[req_id][n_exp]['correction_applied']:
            eval_results = corrected_eval_data_dict[req_id][n_exp]
            eval_results['pred_tables'] = corrected_experiments[req_id][n_exp]['pred_tables']
            eval_results['pred_diff'] = corrected_experiments[req_id][n_exp]['pred_diff']
            
            final_eval_data.append(eval_results)
        else:
            eval_results = eval_exp
            eval_results['pred_tables'] = experiments[req_id][n_exp]['pred_tables']
            eval_results['pred_diff'] = experiments[req_id][n_exp]['pred_diff']
            final_eval_data.append(eval_results)
            
    aggregate_metrics = metrics_aggregation(final_eval_data)
    # Save the aggregated metrics to a JSON file
    evaluation_result = {
        "detailed_results": final_eval_data,
        "aggregate_metrics": aggregate_metrics,
        "metadata": eval_data['metadata'],
        "metadata_corrected": corrected_eval_data['metadata'],
    }
    
    return evaluation_result



def get_evaluation_stats(eval_results):
    """
    Generate a concise summary of evaluation statistics.
    
    Args:
        eval_results (dict): Evaluation results from evaluate_alerce_queries
    
    Returns:
        dict: A dictionary of key evaluation metrics
    """
    metrics = eval_results['aggregate_metrics']
    meta = eval_results['metadata']
    
    total_evaluable_overall = metrics['errors']['total_processed_after_gold_errors']
    
    summary = {
        'total_queries': meta['total_queries'],
        'evaluated_queries': meta['evaluated_queries'],
        'skipped_queries': meta['skipped_queries'],
        'execution_time': meta['evaluation_time'],
        'execution_method': meta.get('execution_method', 'unknown'),
        
        'oids_precision': metrics['oids']['precision'],
        'oids_recall': metrics['oids']['recall'],
        'oids_f1': metrics['oids']['f1_score'],
        'oids_perfect_match_rate': metrics['oids'].get('perfect_match_rate', 0),
        'oids_perfect_match_count': metrics['oids'].get('perfect_match_count', 0),
        'oids_success_rate': metrics['oids']['success_count'] / total_evaluable_overall if total_evaluable_overall > 0 else 0,
        
        'columns_precision': metrics['columns']['precision'],
        'columns_recall': metrics['columns']['recall'],
        'columns_f1': metrics['columns']['f1_score'],
        'columns_perfect_match_rate': metrics['columns'].get('perfect_match_rate', 0),
        'columns_perfect_match_count': metrics['columns'].get('perfect_match_count', 0),
        'columns_success_rate': metrics['columns']['success_count'] / total_evaluable_overall if total_evaluable_overall > 0 else 0,

        'columns_formatted_precision': metrics['columns_formatted']['precision'],
        'columns_formatted_recall': metrics['columns_formatted']['recall'],
        'columns_formatted_f1': metrics['columns_formatted']['f1_score'],
        'columns_formatted_perfect_match_rate': metrics['columns_formatted'].get('perfect_match_rate', 0),
        'columns_formatted_perfect_match_count': metrics['columns_formatted'].get('perfect_match_count', 0),
        'columns_formatted_success_rate': metrics['columns_formatted']['success_count'] / total_evaluable_overall if total_evaluable_overall > 0 else 0,
        
        'error_rate': (metrics['errors']['gold_errors'] + metrics['errors']['pred_errors']) / meta['evaluated_queries'] if meta['evaluated_queries'] > 0 else 1.0,
        'gold_errors': metrics['errors']['gold_errors'],
        'pred_errors': metrics['errors']['pred_errors'],
    }

    # Add by_n_run statistics
    if 'by_n_run' in metrics:
        summary['by_n_run'] = {}
        for n_run_key, n_run_metrics in metrics['by_n_run'].items():
            total_evaluable_in_group = n_run_metrics['oids']['group_total_items'] - n_run_metrics['errors']['gold']
            summary['by_n_run'][n_run_key] = {
                'oids_precision': n_run_metrics['oids']['precision'],
                'oids_recall': n_run_metrics['oids']['recall'],
                'oids_f1': n_run_metrics['oids']['f1_score'],
                'oids_perfect_match_rate': n_run_metrics['oids'].get('perfect_match_rate', 0),
                'oids_perfect_match_count': n_run_metrics['oids'].get('perfect_match_count', 0),
                'oids_success_rate': n_run_metrics['oids']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_precision': n_run_metrics['columns']['precision'],
                'columns_recall': n_run_metrics['columns']['recall'],
                'columns_f1': n_run_metrics['columns']['f1_score'],
                'columns_perfect_match_rate': n_run_metrics['columns'].get('perfect_match_rate', 0),
                'columns_perfect_match_count': n_run_metrics['columns'].get('perfect_match_count', 0),
                'columns_success_rate': n_run_metrics['columns']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_formatted_precision': n_run_metrics['columns_formatted']['precision'],
                'columns_formatted_recall': n_run_metrics['columns_formatted']['recall'],
                'columns_formatted_f1': n_run_metrics['columns_formatted']['f1_score'],
                'columns_formatted_perfect_match_rate': n_run_metrics['columns_formatted'].get('perfect_match_rate', 0),
                'columns_formatted_perfect_match_count': n_run_metrics['columns_formatted'].get('perfect_match_count', 0),
                'columns_formatted_success_rate': n_run_metrics['columns_formatted']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'error_rate': (n_run_metrics['errors']['gold'] + n_run_metrics['errors']['pred']) / n_run_metrics['group_total_items'] if n_run_metrics['group_total_items'] > 0 else 1.0,
                'gold_errors': n_run_metrics['errors']['gold'],
                'pred_errors': n_run_metrics['errors']['pred'],
                'total_items_in_group': n_run_metrics['group_total_items'],
                'total_evaluable_in_group': total_evaluable_in_group
            }

    # Add by_req_id statistics
    if 'by_req_id' in metrics:
        summary['by_req_id'] = {}
        for req_id, req_metrics in metrics['by_req_id'].items():
            total_evaluable_in_group = req_metrics['oids']['group_total_items'] - req_metrics['errors']['gold']
            summary['by_req_id'][req_id] = {
                'oids_precision': req_metrics['oids']['precision'],
                'oids_recall': req_metrics['oids']['recall'],
                'oids_f1': req_metrics['oids']['f1_score'],
                'oids_perfect_match_rate': req_metrics['oids'].get('perfect_match_rate', 0),
                'oids_perfect_match_count': req_metrics['oids'].get('perfect_match_count', 0),
                'oids_success_rate': req_metrics['oids']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_precision': req_metrics['columns']['precision'],
                'columns_recall': req_metrics['columns']['recall'],
                'columns_f1': req_metrics['columns']['f1_score'],
                'columns_perfect_match_rate': req_metrics['columns'].get('perfect_match_rate', 0),
                'columns_perfect_match_count': req_metrics['columns'].get('perfect_match_count', 0),
                'columns_success_rate': req_metrics['columns']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_formatted_precision': req_metrics['columns_formatted']['precision'],
                'columns_formatted_recall': req_metrics['columns_formatted']['recall'],
                'columns_formatted_f1': req_metrics['columns_formatted']['f1_score'],
                'columns_formatted_perfect_match_rate': req_metrics['columns_formatted'].get('perfect_match_rate', 0),
                'columns_formatted_perfect_match_count': req_metrics['columns_formatted'].get('perfect_match_count', 0),
                'columns_formatted_success_rate': req_metrics['columns_formatted']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'error_rate': (req_metrics['errors']['gold'] + req_metrics['errors']['pred']) / req_metrics['group_total_items'] if req_metrics['group_total_items'] > 0 else 1.0,
                'gold_errors': req_metrics['errors']['gold'],
                'pred_errors': req_metrics['errors']['pred'],
                'total_items_in_group': req_metrics['group_total_items'],
                'total_evaluable_in_group': total_evaluable_in_group
            }

    # Add by_difficulty statistics
    if 'by_difficulty' in metrics:
        summary['by_difficulty'] = {}
        for diff_key, diff_metrics in metrics['by_difficulty'].items():
            total_evaluable_in_group = diff_metrics['oids']['group_total_items'] - diff_metrics['errors']['gold']
            summary['by_difficulty'][diff_key] = {
                'oids_precision': diff_metrics['oids']['precision'],
                'oids_recall': diff_metrics['oids']['recall'],
                'oids_f1': diff_metrics['oids']['f1_score'],
                'oids_perfect_match_rate': diff_metrics['oids'].get('perfect_match_rate', 0),
                'oids_perfect_match_count': diff_metrics['oids'].get('perfect_match_count', 0),
                'oids_success_rate': diff_metrics['oids']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_precision': diff_metrics['columns']['precision'],
                'columns_recall': diff_metrics['columns']['recall'],
                'columns_f1': diff_metrics['columns']['f1_score'],
                'columns_perfect_match_rate': diff_metrics['columns'].get('perfect_match_rate', 0),
                'columns_perfect_match_count': diff_metrics['columns'].get('perfect_match_count', 0),
                'columns_success_rate': diff_metrics['columns']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'columns_formatted_precision': diff_metrics['columns_formatted']['precision'],
                'columns_formatted_recall': diff_metrics['columns_formatted']['recall'],
                'columns_formatted_f1': diff_metrics['columns_formatted']['f1_score'],
                'columns_formatted_perfect_match_rate': diff_metrics['columns_formatted'].get('perfect_match_rate', 0),
                'columns_formatted_perfect_match_count': diff_metrics['columns_formatted'].get('perfect_match_count', 0),
                'columns_formatted_success_rate': diff_metrics['columns_formatted']['success_count'] / total_evaluable_in_group if total_evaluable_in_group > 0 else 0,
                'error_rate': (diff_metrics['errors']['gold'] + diff_metrics['errors']['pred']) / diff_metrics['group_total_items'] if diff_metrics['group_total_items'] > 0 else 1.0,
                'gold_errors': diff_metrics['errors']['gold'],
                'pred_errors': diff_metrics['errors']['pred'],
                'total_items_in_group': diff_metrics['group_total_items'],
                'total_evaluable_in_group': total_evaluable_in_group
            }
    
    return summary

def compare_evaluations(sequential_results, parallel_results):
    """
    Compare sequential and parallel evaluation results to verify consistency.
    
    Args:
        sequential_results (dict): Results from sequential evaluation
        parallel_results (dict): Results from parallel evaluation
        
    Returns:
        dict: A dictionary of comparison metrics
    """
    seq_metrics = sequential_results['aggregate_metrics']
    par_metrics = parallel_results['aggregate_metrics']
    seq_meta = sequential_results['metadata']
    par_meta = parallel_results['metadata']
    
    # Compare key metrics
    comparison = {
        'oids_f1_diff': par_metrics['oids']['f1_score'] - seq_metrics['oids']['f1_score'],
        'columns_f1_diff': par_metrics['columns']['f1_score'] - seq_metrics['columns']['f1_score'],
        'oids_perfect_match_diff': par_metrics['oids'].get('perfect_match_rate', 0) - seq_metrics['oids'].get('perfect_match_rate', 0),
        'columns_perfect_match_diff': par_metrics['columns'].get('perfect_match_rate', 0) - seq_metrics['columns'].get('perfect_match_rate', 0),
        'sequential_time': seq_meta['evaluation_time'],
        'parallel_time': par_meta['evaluation_time'],
        'speedup': seq_meta['evaluation_time'] / par_meta['evaluation_time'] if par_meta['evaluation_time'] > 0 else float('inf'),
        'parallel_processes': par_meta.get('parallel_processes', 'unknown'),
        'consistency': 'identical' if (
            par_metrics['oids']['f1_score'] == seq_metrics['oids']['f1_score'] and
            par_metrics['columns']['f1_score'] == seq_metrics['columns']['f1_score'] and
            par_metrics['oids'].get('perfect_match_rate', 0) == seq_metrics['oids'].get('perfect_match_rate', 0) and
            par_metrics['columns'].get('perfect_match_rate', 0) == seq_metrics['columns'].get('perfect_match_rate', 0) and
            par_metrics['errors'] == seq_metrics['errors']
        ) else 'different'
    }
    
    return comparison


def load_evaluation_results(
        save_path: str,
        model_name: str,
        exp_name: str,
) -> dict:
    """
    Load evaluation results from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing evaluation results.
        
    Returns:
        dict: A dictionary containing the evaluation results.
    """
    
    # Construct the file path for the evaluation results
    exp_file_path = os.path.join(save_path, model_name, exp_name, f"{exp_name}.json")
    eval_file_path = os.path.join(save_path, model_name, exp_name, f"eval_{exp_name}.json")
    
    # Check if the file exists
    if not os.path.exists(eval_file_path):
        raise FileNotFoundError(f"Evaluation results file not found: {eval_file_path}")
    
    # Load the experiment data from the JSON file
    with open(exp_file_path, 'r') as f:
        experiments = json.load(f)
    # Load the evaluation results from the JSON file
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)
    eval_data_results = eval_data.get('detailed_results')

    final_eval_data = []
    for eval_exp in eval_data_results:
        req_id = eval_exp['req_id']
        n_exp = eval_exp['n_exp']
        eval_results = eval_exp
        eval_results['pred_tables'] = experiments[req_id][n_exp]['pred_tables']
        eval_results['pred_diff'] = experiments[req_id][n_exp]['pred_diff']
        final_eval_data.append(eval_results)
    # Save the aggregated metrics to a JSON file
    evaluation_result = {
        "detailed_results": final_eval_data,
        "aggregate_metrics": eval_data['aggregate_metrics'],
        "metadata": eval_data['metadata']
    }
    
    return evaluation_result

def load_evaluation_results_from_dir(
        save_path: str,
        dataset = None,
        sql_gen_method = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Load evaluation results from a directory.
    
    Args:
        save_path (str): Path to the directory containing evaluation results.

    Returns:
        Dict[str, Dict[str, dict]]: A dictionary where keys are model names and values are dictionaries
                                    of experiment names and their corresponding evaluation results.
    """
    # Iterate through all folders in the save_path (assumed to be the model name)
    eval_data = {}
    for model_dir in os.listdir(save_path):
        model_name = os.path.basename(model_dir)
        model_path = os.path.join(save_path, model_name)
        eval_data[str(model_name)] = {}
        # Iterate through all experiment folders (assumed to be exp_name)
        for exp_name in os.listdir(model_path):
            if exp_name.startswith('schema_linking') or exp_name.startswith('difficulty_classification'):
                continue
            exp_path = os.path.join(model_path, exp_name)
            if not os.path.exists(os.path.join(exp_path, 'config.json')):
                print(f"Skipping {model_name}/{exp_name} as config.json does not exist.")
                continue
            with open(os.path.join(exp_path, 'config.json'), 'r') as f:
                config = json.load(f)
            if dataset:
                if os.path.basename(config.get('data_path')) != dataset+"."+config.get('data_format'):
                    continue
            if sql_gen_method:
                if config.get('sql_gen_method') != sql_gen_method:
                    continue

            # Check if the self-corrected evaluation file exists
            corrected_eval_file_path = os.path.join(exp_path, f"eval_corrected_{exp_name}.json")
            eval_data[str(model_name)][str(exp_name)] = {}
            if os.path.exists(corrected_eval_file_path):
                try:
                    eval_data[str(model_name)][str(exp_name)]['self_corrected'] = {}
                    eval_data_exp = join_eval_results(
                        save_path=save_path,
                        model_name=model_name,
                        exp_name=exp_name
                    )
                    # Append the detailed results to the eval_data
                    eval_data_exp['metadata']['self_corrected'] = True
                    eval_data_exp['metadata']['sql_gen_method'] = config.get('sql_gen_method', 'unknown')
                    eval_data_exp['metadata']['n_exps'] = config.get('n_exps', 'unknown')
                    eval_data[str(model_name)][str(exp_name)]['self_corrected'] = eval_data_exp
                except Exception as e:
                    print(f"Error loading evaluation results for {model_name}/{exp_name}: {e}")
                    
            # Load the non-corrected evaluation results
            try:
                eval_data[str(model_name)][str(exp_name)]['corrected'] = {}
                eval_data_exp = load_evaluation_results(
                    save_path=save_path,
                    model_name=model_name,
                    exp_name=exp_name
                )
                eval_data_exp['metadata']['self_corrected'] = False
                eval_data_exp['metadata']['sql_gen_method'] = config.get('sql_gen_method', 'unknown')
                eval_data_exp['metadata']['n_exps'] = config.get('n_exps', 'unknown')

                # Append the detailed results to the eval_data
                eval_data[str(model_name)][str(exp_name)]['corrected'] = eval_data_exp
            except:
                print(f"Evaluation results file not found for {model_name}/{exp_name}. Skipping...")
                continue # If the file does not exist, skip this experiment

        # if no experiments were found for this model, remove it
        if not eval_data[str(model_name)]:
            del eval_data[str(model_name)]
    return eval_data

import pandas as pd
def evaluation_results_to_dataframe(
        eval_data: Dict[str, Dict[str, dict]],
) -> pd.DataFrame:
    """
    Load evaluation results into a pandas DataFrame.

    Args:
        eval_data (Dict[str, Dict[str, dict]]): The evaluation data to load.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results.
    """
    rows = []
    for model_name, exps in eval_data.items():
        for exp_name, results in exps.items():
            if "self_corrected" in results:
                for row_i in results["self_corrected"]['detailed_results']:
                    sql_output = row_i['comparison']
                    oids_results = sql_output.get('oids')
                    cols_results = sql_output.get('columns')
                    row = {
                        # experiment metadata
                        "model_name": model_name,
                        "exp_name": exp_name,
                        "req_id": row_i["req_id"],
                        "n_exp": row_i["n_exp"],
                        # SQL output metadata
                        "gold_diff": row_i["difficulty"],
                        "pred_diff": row_i["pred_diff"],
                        "pred_tables": row_i["pred_tables"],
                        "gold_sql": sql_output['sql_gold'],
                        "pred_sql": sql_output['sql_pred'],
                        # oid results
                        'oid_precision': oids_results.get('precision', None),
                        'oid_recall': oids_results.get('recall', None),
                        'oid_f1_score': oids_results.get('f1_score', None),
                        'oid_perfect_match': oids_results.get('perfect_match', None),
                        'oid_true_positives': oids_results.get('true_positives', None),
                        'oid_false_positives': oids_results.get('false_positives', None),
                        'oid_false_negatives': oids_results.get('false_negatives', None),
                        'oid_size_gold': oids_results.get('size_gold', None),
                        'oid_size_pred': oids_results.get('size_pred', None),
                        'oid_comparison_type': oids_results.get('comparison_type', None),
                        'oid_gold_id_col': oids_results.get('gold_id_col', None),
                        'oid_pred_id_col': oids_results.get('pred_id_col', None),
                        # column results
                        'cols_precision': cols_results.get('precision', None),
                        'cols_recall': cols_results.get('recall', None),
                        'cols_f1_score': cols_results.get('f1_score', None),
                        'cols_perfect_match': cols_results.get('perfect_match', None),
                        'cols_true_positives': cols_results.get('true_positives', None),
                        'cols_false_positives': cols_results.get('false_positives', None),
                        'cols_false_negatives': cols_results.get('false_negatives', None),
                        'cols_size_gold': cols_results.get('size_gold', None),
                        'cols_size_pred': cols_results.get('size_pred', None),
                        'cols_comparison_type': cols_results.get('comparison_type', None),
                        # column results formatted
                        'cols_precision_formatted': cols_results.get('precision_formatted', None),
                        'cols_recall_formatted': cols_results.get('recall_formatted', None),
                        'cols_f1_score_formatted': cols_results.get('f1_score_formatted', None),
                        'cols_perfect_match_formatted': cols_results.get('perfect_match_formatted', None),
                        'cols_true_positives_formatted': cols_results.get('true_positives_formatted', None),
                        'cols_false_positives_formatted': cols_results.get('false_positives_formatted', None),
                        'cols_false_negatives_formatted': cols_results.get('false_negatives_formatted', None),
                        'cols_size_gold_formatted': cols_results.get('size_gold_formatted', None),
                        'cols_size_pred_formatted': cols_results.get('size_pred_formatted', None),
                        'cols_comparison_type_formatted': cols_results.get('comparison_type_formatted', None),
                        # SQL output results
                        "exec_time_gold": sql_output.get('execution_time_gold', None),
                        "exec_time_pred": sql_output.get('execution_time_pred', None),
                        "error_gold": sql_output.get('error_gold', None),
                        "error_pred": sql_output.get('error_pred', None),
                        "self_corrected": True
                    }
                    rows.append(row)
                
            for row_i in results["corrected"]['detailed_results']:
                sql_output = row_i['comparison']
                oids_results = sql_output.get('oids')
                cols_results = sql_output.get('columns')
                row = {
                    # experiment metadata
                    "model_name": model_name,
                    "exp_name": exp_name,
                    "req_id": row_i["req_id"],
                    "n_exp": row_i["n_exp"],
                    # SQL output metadata
                    "gold_diff": row_i["difficulty"],
                    "pred_diff": row_i["pred_diff"],
                    "pred_tables": row_i["pred_tables"],
                    "gold_sql": sql_output['sql_gold'],
                    "pred_sql": sql_output['sql_pred'],
                    # oid results
                    'oid_precision': oids_results.get('precision', None),
                    'oid_recall': oids_results.get('recall', None),
                    'oid_f1_score': oids_results.get('f1_score', None),
                    'oid_perfect_match': oids_results.get('perfect_match', None),
                    'oid_true_positives': oids_results.get('true_positives', None),
                    'oid_false_positives': oids_results.get('false_positives', None),
                    'oid_false_negatives': oids_results.get('false_negatives', None),
                    'oid_size_gold': oids_results.get('size_gold', None),
                    'oid_size_pred': oids_results.get('size_pred', None),
                    'oid_comparison_type': oids_results.get('comparison_type', None),
                    'oid_gold_id_col': oids_results.get('gold_id_col', None),
                    'oid_pred_id_col': oids_results.get('pred_id_col', None),
                    # column results
                    'cols_precision': cols_results.get('precision', None),
                    'cols_recall': cols_results.get('recall', None),
                    'cols_f1_score': cols_results.get('f1_score', None),
                    'cols_perfect_match': cols_results.get('perfect_match', None),
                    'cols_true_positives': cols_results.get('true_positives', None),
                    'cols_false_positives': cols_results.get('false_positives', None),
                    'cols_false_negatives': cols_results.get('false_negatives', None),
                    'cols_size_gold': cols_results.get('size_gold', None),
                    'cols_size_pred': cols_results.get('size_pred', None),
                    'cols_comparison_type': cols_results.get('comparison_type', None),
                    # column results formatted
                    'cols_precision_formatted': cols_results.get('precision_formatted', None),
                    'cols_recall_formatted': cols_results.get('recall_formatted', None),
                    'cols_f1_score_formatted': cols_results.get('f1_score_formatted', None),
                    'cols_perfect_match_formatted': cols_results.get('perfect_match_formatted', None),
                    'cols_true_positives_formatted': cols_results.get('true_positives_formatted', None),
                    'cols_false_positives_formatted': cols_results.get('false_positives_formatted', None),
                    'cols_false_negatives_formatted': cols_results.get('false_negatives_formatted', None),
                    'cols_size_gold_formatted': cols_results.get('size_gold_formatted', None),
                    'cols_size_pred_formatted': cols_results.get('size_pred_formatted', None),
                    'cols_comparison_type_formatted': cols_results.get('comparison_type_formatted', None),
                    # SQL output results
                    "exec_time_gold": sql_output.get('execution_time_gold', None),
                    "exec_time_pred": sql_output.get('execution_time_pred', None),
                    "error_gold": sql_output.get('error_gold', None),
                    "error_pred": sql_output.get('error_pred', None),
                    "self_corrected": False
                }
                rows.append(row)
    return pd.DataFrame(rows)