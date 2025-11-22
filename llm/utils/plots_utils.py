from typing import Dict, List, Any, Tuple, Union
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ipywidgets import interact, widgets, HBox, VBox
from IPython.display import display, clear_output
import warnings

def plot_perfect_match(
        evaluation_results: Dict[str, Dict[str, Any]],
        self_corr: bool = True
) -> None:
    """
    Plots the perfect match rates for OID and column matches from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
                                   
    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float}
                        }
                    }
                }
            }
        }
    """
    # Extract model and experiment names from the results dictionary
    models = list(evaluation_results.keys())
    
    # Initialize data collection lists
    oid_match_rates: List[float] = []
    column_match_rates: List[float] = []
    experiment_labels: List[str] = []

    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    # Collect data for plotting
    for model, experiments in evaluation_results.items():
        for experiment, content in experiments.items():
            # Extract metrics for this model and experiment
            try:
                results = content[self_corr_key]['aggregate_metrics']
                oid_match_rates.append(results['oids']['perfect_match_rate'])
                column_match_rates.append(results['columns']['perfect_match_rate'])
                experiment_labels.append(f"{model} - {experiment}")
            except KeyError as e:
                print(f"Warning: Missing data for {model} - {experiment}: {e}")

    x = np.arange(len(experiment_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, oid_match_rates, width, label='OID Matches')
    bars2 = ax.bar(x + width/2, column_match_rates, width, label='Column Matches')
    # x axis rotation
    plt.xticks(rotation=85, ha='right')
    ax.set_xlabel('Experiments',)

    ax.set_ylabel('Perfect Match Percentage')
    ax.set_title('Perfect Match Comparison by Model and Experiment')
    ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    # Add values on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_labels)
    ax.legend()

    plt.show()

def plot_perfect_match_by_model(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        self_corr: bool = True
) -> None:
    """
    Plots the perfect match rates for OID and column matches from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.

    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float}
                        }
                    }
                }
            }
        }
    """
    # Extract model and experiment names from the results dictionary
    experiments = list(evaluation_results[model_name].keys())

    # Initialize data collection lists
    oid_match_rates: List[float] = []
    column_match_rates: List[float] = []
    experiment_labels: List[str] = []

    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    # Collect data for plotting
    for experiment in experiments:
        # Extract metrics for this model and experiment
        try:
            results = evaluation_results[model_name][experiment][self_corr_key]['aggregate_metrics']
            oid_match_rates.append(results['oids']['perfect_match_rate'])
            column_match_rates.append(results['columns']['perfect_match_rate'])
            experiment_labels.append(f"{model_name} - {experiment}")
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")

    x = np.arange(len(experiments))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, oid_match_rates, width, label='OID Matches')
    bars2 = ax.bar(x + width/2, column_match_rates, width, label='Column Matches')
    # x axis rotation
    plt.xticks(rotation=85, ha='right')
    ax.set_xlabel('Experiments',)

    ax.set_ylabel('Perfect Match Percentage')
    ax.set_title('Perfect Match Comparison by Experiment for Model: ' + model_name)
    ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    # Add values on top of the bars
    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.legend()

    plt.show()

def plot_metrics(
        evaluation_results: Dict[str, Dict[str, Any]],
        self_corr: bool = True
) -> None:
    """
    Plot precision, recall, and F1 score for each model and experiment, by OID and columns.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
    Returns:
        None: This function displays a plot but does not return any value.
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'precision': float, 'recall': float, 'f1_score': float},
                            'columns': {'precision': float, 'recall': float, 'f1_score': float}
                        }
                    }
                }
            }
        }
    """
    # Extract model and experiment names from the results dictionary
    # models = list(evaluation_results.keys())
    # experiments = list(evaluation_results[models[0]].keys())

    # Initialize data collection lists
    oid_precision: List[float] = []
    oid_recall: List[float] = []
    oid_f1: List[float] = []
    column_precision: List[float] = []
    column_recall: List[float] = []
    column_f1: List[float] = []
    experiment_labels: List[str] = []

    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    # # Collect data for plotting
    # for model in models:
    #     for experiment in experiments:

    for model, experiments in evaluation_results.items():
        print(f"\n========== Model: {model} ========== ")
        for experiment, content in experiments.items():
            print(f"Experiment: {experiment}")
            try:
                results = content[self_corr_key]['aggregate_metrics']
                oid_precision.append(results['oids']['precision'])
                oid_recall.append(results['oids']['recall'])
                oid_f1.append(results['oids']['f1_score'])
                column_precision.append(results['columns']['precision'])
                column_recall.append(results['columns']['recall'])
                column_f1.append(results['columns']['f1_score'])
                experiment_labels.append(f"{model} - {experiment}")
            except KeyError as e:
                print(f"Warning: Missing data for {model} - {experiment}: {e}")

    # make 3 subplots for precision, recall, and f1
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    metrics = ['precision', 'recall', 'f1_score']
    data = [oid_precision, oid_recall, oid_f1, column_precision, column_recall, column_f1]
    labels = ['OID Precision', 'OID Recall', 'OID F1', 'Column Precision', 'Column Recall', 'Column F1']
    width = 0.35  # the width of the bars
    x = np.arange(len(experiment_labels))  # the label locations
    for i, metric in enumerate(metrics):
        axs[i].bar(x - width/2, data[i*2], width, label='OID')
        axs[i].bar(x + width/2, data[i*2 + 1], width, label='Column', )
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_ylim(0, 1)
        axs[i].legend()
        axs[i].tick_params(axis='x', rotation=85)
    # add values on top of the bars
    for i, ax in enumerate(axs):
        for j, v in enumerate(data[i*2]):
            ax.text(j - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        for j, v in enumerate(data[i*2 + 1]):
            ax.text(j + width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(experiment_labels)
    # axs[-1].set_ylabel('Metrics')
    # axs[-1].set_title('Metrics by Model and Experiment')
    axs[-1].set_xlabel('Experiments')
    plt.suptitle('Metrics Comparison by Model and Experiment', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_metrics_by_model(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        self_corr: bool = True
) -> None:
    """
    Plot precision, recall, and F1 score for a specific model across all experiments.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.

    Returns:
        None: This function displays a plot but does not return any value.
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'precision': float, 'recall': float, 'f1_score': float},
                            'columns': {'precision': float, 'recall': float, 'f1_score': float}
                        }
                    }
                }
            }
        }
    """
    # Extract experiment names for the specified model
    experiments = list(evaluation_results[model_name].keys())

    # Initialize data collection lists
    oid_precision: List[float] = []
    oid_recall: List[float] = []
    oid_f1: List[float] = []
    column_precision: List[float] = []
    column_recall: List[float] = []
    column_f1: List[float] = []
    experiment_labels: List[str] = []

    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    # Collect data for plotting
    for experiment in experiments:
        try:
            results = evaluation_results[model_name][experiment][self_corr_key]['aggregate_metrics']
            oid_precision.append(results['oids']['precision'])
            oid_recall.append(results['oids']['recall'])
            oid_f1.append(results['oids']['f1_score'])
            column_precision.append(results['columns']['precision'])
            column_recall.append(results['columns']['recall'])
            column_f1.append(results['columns']['f1_score'])
            experiment_labels.append(f"{model_name} - {experiment}")
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")

    # make 3 subplots for precision, recall, and f1
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    metrics = ['precision', 'recall', 'f1_score']
    data = [oid_precision, oid_recall, oid_f1, column_precision, column_recall, column_f1]
    labels = ['OID Precision', 'OID Recall', 'OID F1', 'Column Precision', 'Column Recall', 'Column F1']
    width = 0.35  # the width of the bars
    x = np.arange(len(experiment_labels))  # the label locations
    for i, metric in enumerate(metrics):
        axs[i].bar(x - width/2, data[i*2], width, label='OID')
        axs[i].bar(x + width/2, data[i*2 + 1], width, label='Column', )
        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_ylim(0, 1)
        axs[i].legend()
        axs[i].tick_params(axis='x', rotation=85)
    # add values on top of the bars
    for i, ax in enumerate(axs):
        for j, v in enumerate(data[i*2]):
            ax.text(j - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
        for j, v in enumerate(data[i*2 + 1]):
            ax.text(j + width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(experiment_labels)
    axs[-1].set_ylabel('Metrics')
    axs[-1].set_title('Metrics by Experiment for Model: ' + model_name)
    axs[-1].set_xlabel('Experiments')
    plt.suptitle('Metrics Comparison for Model: ' + model_name, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_perfect_match_by_difficulty(
        evaluation_results_: Dict[str, Dict[str, Any]],
        std_dev: Union[str, None] = None,
        self_corr: bool = True,
        formatted_columns: bool = True,
        medium_hard: bool = False
) -> None:
    """
    Plots the perfect match rates for OID and column matches by difficulty from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float},
                            'difficulty': str  # e.g., 'easy', 'medium', 'hard'
                        }
                    }
                }
            }
        }
    """
    import copy
    # evaluation_results = evaluation_results_.copy()
    evaluation_results = copy.deepcopy(evaluation_results_)

    # Extract model and experiment names from the results dictionary
    models = sorted(list(evaluation_results.keys()))
    
    # Initialize data collection lists
    experiment_labels = {}
    difficulties = []
    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    from llm.utils.eval_utils import metrics_aggregation

    # Collect data for plotting
    for model in models:
        experiments = sorted(list(evaluation_results[model].keys()))
        for experiment in experiments:
            try:
                exp_label = f"{model}-{experiment}"
                oid_match_rates = {}
                column_match_rates = {}
                if std_dev:
                    oid_match_rates_std = {}
                    column_match_rates_std = {}
                    
                results = evaluation_results[model][experiment][self_corr_key]['detailed_results'].copy()
                if medium_hard:
                    # Change medium and advanced difficulties to medium-hard
                    for res_i in results:
                        if res_i['difficulty'] == 'medium' or res_i['difficulty'] == 'advanced': res_i['difficulty'] = 'medium-hard'
                            
                aggregate_metrics = metrics_aggregation(results=results)
                # ['errors']['gold_errors']
                print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                # Use the aggregate metrics directly
                if std_dev == 'all' or std_dev is None:
                    # results = evaluation_results[model][experiment][self_corr_key]['aggregate_metrics']['by_difficulty']
                    results = aggregate_metrics['by_difficulty']
                elif std_dev == 'run':
                    results = aggregate_metrics['by_difficulty_runs']
                elif std_dev == 'req_id':
                    results = aggregate_metrics['by_difficulty_req_id']
                else:
                    raise ValueError(f"Unknown standard deviation type: {std_dev}")
                # iterate through difficulties
                for difficulty, metrics in results.items():
                    if difficulty not in difficulties:
                        difficulties.append(difficulty)
                    oid_match_rates[difficulty] = metrics['oids']['perfect_match_rate']
                    if std_dev: oid_match_rates_std[difficulty] = metrics['oids']['perfect_match_rate_std']
                    if formatted_columns and 'columns_formatted' in metrics:
                        column_match_rates[difficulty] = metrics['columns_formatted']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns_formatted']['perfect_match_rate_std']
                    else:
                        column_match_rates[difficulty] = metrics['columns']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns']['perfect_match_rate_std']
                # Add standard deviation to the match rates
                if std_dev:
                    experiment_labels[exp_label] = {
                        'oid': oid_match_rates,
                        'oid_std': oid_match_rates_std,
                        'column': column_match_rates,
                        'column_std': column_match_rates_std
                    }
                else:
                    experiment_labels[exp_label] = {
                        'oid': oid_match_rates,
                        'column': column_match_rates
                    }
            except KeyError as e:
                print(f"Warning: Missing data for {model} - {experiment}: {e}")
    # Figure with two subplots: OID and Column matches
    # Each subplot will be separated by difficulty on the x-axis
    # Will be bar plots for each difficulty by experiment
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # difficulties = list(next(iter(experiment_labels.values()))['oid'].keys())
    # order difficulties: simple, medium, advanced
    difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    x = np.arange(len(difficulties))  # the label locations
    width = 0.05  # the width of the bars
    for i, (exp_label, data) in enumerate(experiment_labels.items()):
        oid_rates = [data['oid'][d] for d in difficulties]
        column_rates = [data['column'][d] for d in difficulties]
        # Plot OID matches
        axs[0].bar(x + (i - 1) * width, oid_rates, width, label=exp_label)
        # If std_dev is used, add error bars
        # Plot Column matches
        axs[1].bar(x + (i - 1) * width, column_rates, width, label=exp_label, )
        if std_dev:
            oid_std = [data['oid_std'][d] for d in difficulties]
            axs[0].errorbar(x + (i - 1) * width, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5)
            column_std = [data['column_std'][d] for d in difficulties]
            axs[1].errorbar(x + (i - 1) * width, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5)
    # Set x-ticks and labels
    axs[0].set_ylabel('Rows', fontsize=20)
    axs[1].set_ylabel('Columns', fontsize=20)
    
    axs[1].set_xticks(x)
    # change advanced to hard
    difficulties = [d.replace('advanced', 'hard') for d in difficulties]
    axs[1].set_xticklabels(difficulties)
    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    # set suptitle
    plt.suptitle('Query Generation Strategies', fontsize=24)
    # Add values on top of the bars
    for ax in axs:
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # Add the value on top of the bar
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
            
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.02)
    plt.tight_layout()
    # plt.xlabel('Difficulties')
    plt.legend(title='Experiments', bbox_to_anchor=(1.02, 0.88),)
    plt.show()

def plot_perfect_match_by_difficulty_interactive(
        evaluation_results_: Dict[str, Dict[str, Any]],
        std_dev: Union[str, None] = None,
        formatted_columns: bool = True,
        take_error_gold: bool = False,
) -> None:
    """
    Interactive plot of perfect match rates for OID and column matches by difficulty from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays an interactive plot but does not return any value.
    """
    import copy
    evaluation_results = copy.deepcopy(evaluation_results_)
    # Extract model names from the results dictionary
    models = sorted(list(evaluation_results.keys()))
    
    # Create widgets for interactive selection
    model_dropdown = widgets.Dropdown(
        options=models,
        description='Model:',
        value=models[0] if models else None,
        layout=widgets.Layout(width='50%')
    )
    
    self_corr_dropdown = widgets.Dropdown(
        options=[('With Self-Correction', True), ('Without Self-Correction', False)],
        description='Self-Correction:',
        value=True,
        layout=widgets.Layout(width='50%')
    )

    med_hard_dropdown = widgets.Dropdown(
        options=[('Simple-Medium-Hard', False), ('Simple-Medium/Hard', True)],
        description='Difficulties:',
        value=False,
        layout=widgets.Layout(width='50%')
    )
    
    output_widget = widgets.Output()
    
    from llm.utils.eval_utils import metrics_aggregation
    
    def update_plot(model, self_corr, medium_hard):
        """Update the plot based on selected model and self-correction setting"""
        with output_widget:
            clear_output(wait=True)
            evaluation_results = copy.deepcopy(evaluation_results_)
            
            # Self-corrected results key
            self_corr_key = 'self_corrected' if self_corr else 'corrected'

            # Medium-hard setting
            medium_hard = medium_hard
            
            # Initialize data collection lists
            experiment_labels = {}
            difficulties = []
            
            # Collect data for plotting
            experiments = sorted(list(evaluation_results[model].keys()))
            for experiment in experiments:
                try:
                    exp_label = f"{model}-{experiment}"
                    oid_match_rates = {}
                    column_match_rates = {}
                    if std_dev:
                        oid_match_rates_std = {}
                        column_match_rates_std = {}
                        
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    if medium_hard:
                        # Change medium and advanced difficulties to medium-hard
                        for res_i in results:
                            if res_i['difficulty'] == 'medium' or res_i['difficulty'] == 'advanced': res_i['difficulty'] = 'medium-hard'
                
                    aggregate_metrics = metrics_aggregation(results=results, take_error_gold=take_error_gold)
                    
                    print("=========================================")
                    print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                    print(f"Gold queries with errors (IDs): {np.unique(aggregate_metrics['errors']['gold_list'])}")

                    # Use the aggregate metrics based on std_dev setting
                    if std_dev == 'all' or std_dev is None:
                        results = aggregate_metrics['by_difficulty']
                    elif std_dev == 'run':
                        results = aggregate_metrics['by_difficulty_runs']
                    elif std_dev == 'req_id':
                        results = aggregate_metrics['by_difficulty_req_id']
                    else:
                        raise ValueError(f"Unknown standard deviation type: {std_dev}")
                    
                    # Iterate through difficulties
                    for difficulty, metrics in results.items():
                        if difficulty not in difficulties:
                            difficulties.append(difficulty)
                        
                        # Extract OID metrics
                        oid_match_rates[difficulty] = metrics['oids']['perfect_match_rate']
                        if std_dev: 
                            oid_match_rates_std[difficulty] = metrics['oids']['perfect_match_rate_std']
                        
                        # Extract column metrics
                        if formatted_columns and 'columns_formatted' in metrics:
                            column_match_rates[difficulty] = metrics['columns_formatted']['perfect_match_rate']
                            if std_dev: 
                                column_match_rates_std[difficulty] = metrics['columns_formatted']['perfect_match_rate_std']
                        else:
                            column_match_rates[difficulty] = metrics['columns']['perfect_match_rate']
                            if std_dev: 
                                column_match_rates_std[difficulty] = metrics['columns']['perfect_match_rate_std']
                    
                    # Add metrics to experiment_labels
                    if std_dev:
                        experiment_labels[exp_label] = {
                            'oid': oid_match_rates,
                            'oid_std': oid_match_rates_std,
                            'column': column_match_rates,
                            'column_std': column_match_rates_std  # Fixed: column_rates_std → column_match_rates_std
                        }
                    else:
                        experiment_labels[exp_label] = {
                            'oid': oid_match_rates,
                            'column': column_match_rates
                        }
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment}: {e}")
            
            # Sort difficulties
            difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
            
            # Create the plot
            fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            x = np.arange(len(difficulties))  # the label locations
            width = 0.8 / len(experiment_labels) if experiment_labels else 0.1  # the width of the bars
            
            # Plot each experiment
            for i, (exp_label, data) in enumerate(experiment_labels.items()):
                offset = (i - len(experiment_labels)/2 + 0.5) * width
                oid_rates = [data['oid'][d] for d in difficulties]
                column_rates = [data['column'][d] for d in difficulties]
                
                if "direct" in exp_label:
                    method_label = "Direct"
                elif "sbs" in exp_label or "step-by-step" in exp_label:
                    method_label = "Step-by-Step"
                else:
                    method_label = exp_label
                # Plot OID matches
                bars1 = axs[0].bar(x + offset, oid_rates, width, label=method_label)

                # Plot Column matches
                bars2 = axs[1].bar(x + offset, column_rates, width, label=method_label)
                
                # Add error bars if std_dev is provided
                if std_dev:
                    oid_std = [data['oid_std'][d] for d in difficulties]
                    axs[0].errorbar(x + offset, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5)
                    
                    column_std = [data['column_std'][d] for d in difficulties]
                    axs[1].errorbar(x + offset, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5)
                
                # Add values on top of the bars
                for bar in bars1:
                    height = bar.get_height()
                    axs[0].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
                
                for bar in bars2:
                    height = bar.get_height()
                    axs[1].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
            
            # Set x-ticks and labels
            axs[0].set_ylabel('Rows', fontsize=20)
            axs[1].set_ylabel('Columns', fontsize=20)
            
            axs[1].set_xticks(x)
            # change advanced to hard
            difficulties_labels = [d.replace('advanced', 'hard') for d in difficulties]
            axs[1].set_xticklabels(difficulties_labels)
            axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
            axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
            
            # Set title and labels
            plt.suptitle(f'Query Generation Strategies for {model}', fontsize=24)
            fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.02)
            
            # Add legend
            # plt.legend(title='Experiments', bbox_to_anchor=(1.02, 0.88))
            # plt.legend(title='Experiments', loc='upper left', bbox_to_anchor=(1.02, 1))
            axs[0].legend(title='Experiments', loc='upper left', bbox_to_anchor=(0.88, 1))
            plt.tight_layout()
            plt.show()
    
    # Create interactive widget
    interact_widget = widgets.interactive(
        update_plot,
        model=model_dropdown,
        self_corr=self_corr_dropdown,
        medium_hard=med_hard_dropdown
    )
    
    # Display the widgets and output
    display(widgets.VBox([interact_widget, output_widget]))
    
    # Initial plot
    if models:
        update_plot(models[0], True, False)

def plot_metrics_by_difficulty(
        evaluation_results: Dict[str, Dict[str, Any]],
        std_dev: Union[str, None] = None,
        self_corr: bool = True,
        formatted_columns: bool = True
) -> None:
    """
    Plots precision, recall, and F1 score metrics by difficulty from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays a plot but does not return any value.
    """
    # Extract model and experiment names from the results dictionary
    models = sorted(list(evaluation_results.keys()))
    
    # Initialize data collection lists
    experiment_labels = {}
    difficulties = []
    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    from llm.utils.eval_utils import metrics_aggregation

    # Collect data for plotting
    for model in models:
        experiments = sorted(list(evaluation_results[model].keys()))
        for experiment in experiments:
            try:
                exp_label = f"{model}-{experiment}"
                
                # Initialize dictionaries for each metric
                oid_precision = {}
                oid_recall = {}
                oid_f1 = {}
                column_precision = {}
                column_recall = {}
                column_f1 = {}
                
                # Initialize standard deviation dictionaries if needed
                if std_dev:
                    oid_precision_std = {}
                    oid_recall_std = {}
                    oid_f1_std = {}
                    column_precision_std = {}
                    column_recall_std = {}
                    column_f1_std = {}
                    
                results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                aggregate_metrics = metrics_aggregation(results=results)
                
                print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                
                # Use the aggregate metrics directly
                if std_dev == 'all' or std_dev is None:
                    results = aggregate_metrics['by_difficulty']
                elif std_dev == 'run':
                    results = aggregate_metrics['by_difficulty_runs']
                elif std_dev == 'req_id':
                    results = aggregate_metrics['by_difficulty_req_id']
                else:
                    raise ValueError(f"Unknown standard deviation type: {std_dev}")
                
                # Iterate through difficulties
                for difficulty, metrics in results.items():
                    if difficulty not in difficulties:
                        difficulties.append(difficulty)
                    
                    # Extract OID metrics
                    oid_precision[difficulty] = metrics['oids']['precision']
                    oid_recall[difficulty] = metrics['oids']['recall']
                    oid_f1[difficulty] = metrics['oids']['f1_score']
                    
                    # Extract column metrics
                    if formatted_columns and 'columns_formatted' in metrics:
                        column_precision[difficulty] = metrics['columns_formatted']['precision']
                        column_recall[difficulty] = metrics['columns_formatted']['recall']
                        column_f1[difficulty] = metrics['columns_formatted']['f1_score']
                    else:
                        column_precision[difficulty] = metrics['columns']['precision']
                        column_recall[difficulty] = metrics['columns']['recall']
                        column_f1[difficulty] = metrics['columns']['f1_score']
                    
                    # Extract standard deviations if needed
                    if std_dev:
                        oid_precision_std[difficulty] = metrics['oids'].get('precision_std', 0)
                        oid_recall_std[difficulty] = metrics['oids'].get('recall_std', 0)
                        oid_f1_std[difficulty] = metrics['oids'].get('f1_score_std', 0)
                        
                        if formatted_columns and 'columns_formatted' in metrics:
                            column_precision_std[difficulty] = metrics['columns_formatted'].get('precision_std', 0)
                            column_recall_std[difficulty] = metrics['columns_formatted'].get('recall_std', 0)
                            column_f1_std[difficulty] = metrics['columns_formatted'].get('f1_score_std', 0)
                        else:
                            column_precision_std[difficulty] = metrics['columns'].get('precision_std', 0)
                            column_recall_std[difficulty] = metrics['columns'].get('recall_std', 0)
                            column_f1_std[difficulty] = metrics['columns'].get('f1_score_std', 0)
                
                # Add metrics to experiment_labels
                if std_dev:
                    experiment_labels[exp_label] = {
                        'oid_precision': oid_precision,
                        'oid_precision_std': oid_precision_std,
                        'oid_recall': oid_recall,
                        'oid_recall_std': oid_recall_std,
                        'oid_f1': oid_f1,
                        'oid_f1_std': oid_f1_std,
                        'column_precision': column_precision,
                        'column_precision_std': column_precision_std,
                        'column_recall': column_recall,
                        'column_recall_std': column_recall_std,
                        'column_f1': column_f1,
                        'column_f1_std': column_f1_std
                    }
                else:
                    experiment_labels[exp_label] = {
                        'oid_precision': oid_precision,
                        'oid_recall': oid_recall,
                        'oid_f1': oid_f1,
                        'column_precision': column_precision,
                        'column_recall': column_recall,
                        'column_f1': column_f1
                    }
            except KeyError as e:
                print(f"Warning: Missing data for {model} - {experiment}: {e}")
    
    # Sort difficulties
    difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    
    # Create a figure with 6 subplots (3 rows, 2 columns)
    # Row 1: Precision (OID, Column)
    # Row 2: Recall (OID, Column)
    # Row 3: F1 Score (OID, Column)
    fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    
    # Set titles for each subplot
    titles = [('Precision (Rows)', 'Precision (Columns)'),
              ('Recall (Rows)', 'Recall (Columns)'),
              ('F1 Score (Rows)', 'F1 Score (Columns)')]
    
    for i, row_titles in enumerate(titles):
        for j, title in enumerate(row_titles):
            axs[i, j].set_title(title, fontsize=14)
    
    # Data mapping for easy access
    data_keys = [
        ['oid_precision', 'column_precision'],
        ['oid_recall', 'column_recall'],
        ['oid_f1', 'column_f1']
    ]
    
    std_keys = [
        ['oid_precision_std', 'column_precision_std'],
        ['oid_recall_std', 'column_recall_std'],
        ['oid_f1_std', 'column_f1_std']
    ]
    
    x = np.arange(len(difficulties))  # the label locations
    width = 0.05  # the width of the bars
    
    # Plot each experiment
    for i, (exp_label, data) in enumerate(experiment_labels.items()):
        for row in range(3):  # 3 rows: precision, recall, f1
            for col in range(2):  # 2 columns: OID, column
                key = data_keys[row][col]
                rates = [data[key][d] for d in difficulties]
                
                # Plot bars
                bars = axs[row, col].bar(x + (i - 1) * width, rates, width, label=exp_label)
                
                # Add error bars if std_dev is used
                if std_dev:
                    std_key = std_keys[row][col]
                    stds = [data[std_key][d] for d in difficulties]
                    axs[row, col].errorbar(x + (i - 1) * width, rates, yerr=stds, fmt='none', ecolor='black', capsize=5)
                
                # Add values on top of the bars
                for bar in bars:
                    height = bar.get_height()
                    axs[row, col].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
    
    # Set common x-axis labels and ticks
    for ax in axs[-1, :]:
        ax.set_xticks(x)
        # change advanced to hard
        difficulties_labels = [d.replace('advanced', 'hard') for d in difficulties]
        ax.set_xticklabels(difficulties_labels)
    
    # Set y-axis limits
    for ax in axs.flatten():
        ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    
    # Add a common legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Experiments', bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    # Set common y-label
    fig.text(0.02, 0.5, 'Score', va='center', rotation='vertical', fontsize=20)
    
    # Set main title
    plt.suptitle('Metrics by Difficulty', fontsize=24)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.show()

def plot_metrics_by_difficulty_interactive(
        evaluation_results: Dict[str, Dict[str, Any]],
        std_dev: Union[str, None] = None,
        formatted_columns: bool = True
) -> None:
    """
    Interactive plot of precision, recall, and F1 score metrics by difficulty from evaluation results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays an interactive plot but does not return any value.
    """
    # Extract model names from the results dictionary
    models = sorted(list(evaluation_results.keys()))
    
    # Create widgets for interactive selection
    model_dropdown = widgets.Dropdown(
        options=models,
        description='Model:',
        value=models[0] if models else None,
        layout=widgets.Layout(width='50%')
    )
    
    self_corr_dropdown = widgets.Dropdown(
        options=[('With Self-Correction', True), ('Without Self-Correction', False)],
        description='Self-Correction:',
        value=True,
        layout=widgets.Layout(width='50%')
    )
    
    output_widget = widgets.Output()
    
    from llm.utils.eval_utils import metrics_aggregation
    
    def update_plot(model, self_corr):
        """Update the plot based on selected model and self-correction setting"""
        with output_widget:
            clear_output(wait=True)
            
            # Self-corrected results key
            self_corr_key = 'self_corrected' if self_corr else 'corrected'
            
            # Initialize data collection lists
            experiment_labels = {}
            difficulties = []
            
            # Collect data for plotting
            experiments = sorted(list(evaluation_results[model].keys()))
            for experiment in experiments:
                try:
                    exp_label = f"{model}-{experiment}"
                    
                    # Initialize dictionaries for each metric
                    oid_precision = {}
                    oid_recall = {}
                    oid_f1 = {}
                    column_precision = {}
                    column_recall = {}
                    column_f1 = {}
                    
                    # Initialize standard deviation dictionaries if needed
                    if std_dev:
                        oid_precision_std = {}
                        oid_recall_std = {}
                        oid_f1_std = {}
                        column_precision_std = {}
                        column_recall_std = {}
                        column_f1_std = {}
                        
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    aggregate_metrics = metrics_aggregation(results=results)
                    
                    print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                    
                    # Use the aggregate metrics based on std_dev setting
                    if std_dev == 'all' or std_dev is None:
                        results = aggregate_metrics['by_difficulty']
                    elif std_dev == 'run':
                        results = aggregate_metrics['by_difficulty_runs']
                    elif std_dev == 'req_id':
                        results = aggregate_metrics['by_difficulty_req_id']
                    else:
                        raise ValueError(f"Unknown standard deviation type: {std_dev}")
                    
                    # Iterate through difficulties
                    for difficulty, metrics in results.items():
                        if difficulty not in difficulties:
                            difficulties.append(difficulty)
                        
                        # Extract OID metrics
                        oid_precision[difficulty] = metrics['oids']['precision']
                        oid_recall[difficulty] = metrics['oids']['recall']
                        oid_f1[difficulty] = metrics['oids']['f1_score']
                        
                        # Extract column metrics
                        if formatted_columns and 'columns_formatted' in metrics:
                            column_precision[difficulty] = metrics['columns_formatted']['precision']
                            column_recall[difficulty] = metrics['columns_formatted']['recall']
                            column_f1[difficulty] = metrics['columns_formatted']['f1_score']
                        else:
                            column_precision[difficulty] = metrics['columns']['precision']
                            column_recall[difficulty] = metrics['columns']['recall']
                            column_f1[difficulty] = metrics['columns']['f1_score']
                        
                        # Extract standard deviations if needed
                        if std_dev:
                            oid_precision_std[difficulty] = metrics['oids'].get('precision_std', 0)
                            oid_recall_std[difficulty] = metrics['oids'].get('recall_std', 0)
                            oid_f1_std[difficulty] = metrics['oids'].get('f1_score_std', 0)
                            
                            if formatted_columns and 'columns_formatted' in metrics:
                                column_precision_std[difficulty] = metrics['columns_formatted'].get('precision_std', 0)
                                column_recall_std[difficulty] = metrics['columns_formatted'].get('recall_std', 0)
                                column_f1_std[difficulty] = metrics['columns_formatted'].get('f1_score_std', 0)
                            else:
                                column_precision_std[difficulty] = metrics['columns'].get('precision_std', 0)
                                column_recall_std[difficulty] = metrics['columns'].get('recall_std', 0)
                                column_f1_std[difficulty] = metrics['columns'].get('f1_score_std', 0)
                    
                    # Add metrics to experiment_labels
                    if std_dev:
                        experiment_labels[exp_label] = {
                            'oid_precision': oid_precision,
                            'oid_precision_std': oid_precision_std,
                            'oid_recall': oid_recall,
                            'oid_recall_std': oid_recall_std,
                            'oid_f1': oid_f1,
                            'oid_f1_std': oid_f1_std,
                            'column_precision': column_precision,
                            'column_precision_std': column_precision_std,
                            'column_recall': column_recall,
                            'column_recall_std': column_recall_std,
                            'column_f1': column_f1,
                            'column_f1_std': column_f1_std
                        }
                    else:
                        experiment_labels[exp_label] = {
                            'oid_precision': oid_precision,
                            'oid_recall': oid_recall,
                            'oid_f1': oid_f1,
                            'column_precision': column_precision,
                            'column_recall': column_recall,
                            'column_f1': column_f1
                        }
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment}: {e}")
            
            # Sort difficulties
            difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
            
            # Create a figure with 6 subplots (3 rows, 2 columns)
            fig, axs = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
            
            # Set titles for each subplot
            titles = [('Precision (Rows)', 'Precision (Columns)'),
                      ('Recall (Rows)', 'Recall (Columns)'),
                      ('F1 Score (Rows)', 'F1 Score (Columns)')]
            
            for i, row_titles in enumerate(titles):
                for j, title in enumerate(row_titles):
                    axs[i, j].set_title(title, fontsize=14)
            
            # Data mapping for easy access
            data_keys = [
                ['oid_precision', 'column_precision'],
                ['oid_recall', 'column_recall'],
                ['oid_f1', 'column_f1']
            ]
            
            std_keys = [
                ['oid_precision_std', 'column_precision_std'],
                ['oid_recall_std', 'column_recall_std'],
                ['oid_f1_std', 'column_f1_std']
            ]
            
            x = np.arange(len(difficulties))  # the label locations
            width = 0.8 / len(experiment_labels) if experiment_labels else 0.1  # the width of the bars
            
            # Plot each experiment
            for i, (exp_label, data) in enumerate(experiment_labels.items()):
                offset = (i - len(experiment_labels)/2 + 0.5) * width
                
                for row in range(3):  # 3 rows: precision, recall, f1
                    for col in range(2):  # 2 columns: OID, column
                        key = data_keys[row][col]
                        rates = [data[key][d] for d in difficulties]
                        
                        # Plot bars
                        bars = axs[row, col].bar(x + offset, rates, width, label=exp_label)
                        
                        # Add error bars if std_dev is used
                        if std_dev:
                            std_key = std_keys[row][col]
                            stds = [data[std_key][d] for d in difficulties]
                            axs[row, col].errorbar(x + offset, rates, yerr=stds, fmt='none', ecolor='black', capsize=5)
                        
                        # Add values on top of the bars
                        for bar in bars:
                            height = bar.get_height()
                            axs[row, col].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
    
            # Set common x-axis labels and ticks
            for ax in axs[-1, :]:
                ax.set_xticks(x)
                # change advanced to hard
                difficulties_labels = [d.replace('advanced', 'hard') for d in difficulties]
                ax.set_xticklabels(difficulties_labels)
            
            # Set y-axis limits
            for ax in axs.flatten():
                ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
            
            # Add a common legend
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, title='Experiments', bbox_to_anchor=(1.02, 0.5), loc='center left')
            
            # Set common y-label
            fig.text(0.02, 0.5, 'Score', va='center', rotation='vertical', fontsize=20)
    
            # Set main title
            plt.suptitle('Metrics by Difficulty', fontsize=24)
            
            plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
            plt.subplots_adjust(hspace=0.3, wspace=0.2)
            plt.show()

    # Create interactive widget
    interact_widget = widgets.interactive(
        update_plot,
        model=model_dropdown,
        self_corr=self_corr_dropdown
    )
    
    # Display the widgets and output
    display(widgets.VBox([interact_widget, output_widget]))
    
    # Initial plot
    if models:
        update_plot(models[0], True)

def plot_perfect_match_by_difficulty_model(
        evaluation_results_: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: List[str],
        std_dev: Union[str, None] = None,
        self_corr: bool = True,
        formatted_columns: bool = True,
        medium_hard: bool = False,
        take_error_gold: bool = False,
        save_fig: Union[str, None] = None
) -> None:
    """
    Plots the perfect match rates for OID and column matches by difficulty from evaluation results for a specific model and experiments.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_names (List[str]): A list of experiment names to include in the plot.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float},
                            'difficulty': str  # e.g., 'easy', 'medium', 'hard'
                        }
                    }
                }
            }
        }
    """
    import copy
    evaluation_results = copy.deepcopy(evaluation_results_)
    # Initialize data collection lists
    experiment_labels = {}
    difficulties = []
    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    from llm.utils.eval_utils import metrics_aggregation

    # Collect data for plotting
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    for experiment in experiments:
        try:
            if "dir" in experiment:
                exp_label = "Direct"
            if "sbs" in experiment:
                exp_label = "Step-by-Step"
            
            oid_match_rates = {}
            column_match_rates = {}
            if std_dev:
                oid_match_rates_std = {}
                column_match_rates_std = {}

            results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
            if medium_hard:
                # Change medium and advanced difficulties to medium-hard
                for res_i in results:
                    if res_i['difficulty'] == 'medium' or res_i['difficulty'] == 'advanced': res_i['difficulty'] = 'medium-hard'
                
            aggregate_metrics = metrics_aggregation(results=results, take_error_gold=take_error_gold)
            # ['errors']['gold_errors']
            print("================================")
            print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
            print(f"Gold queries with errors (IDs): {np.unique(aggregate_metrics['errors']['gold_list'])}")
            # Use the aggregate metrics directly
            if std_dev == 'all' or std_dev is None:
                # results = evaluation_results[model][experiment][self_corr_key]['aggregate_metrics']['by_difficulty']
                results = aggregate_metrics['by_difficulty']
            elif std_dev == 'run':
                results = aggregate_metrics['by_difficulty_runs']
            elif std_dev == 'req_id':
                results = aggregate_metrics['by_difficulty_req_id']
            else:
                raise ValueError(f"Unknown standard deviation type: {std_dev}")
            # iterate through difficulties
            for difficulty, metrics in results.items():
                if difficulty not in difficulties:
                    difficulties.append(difficulty)
                oid_match_rates[difficulty] = metrics['oids']['perfect_match_rate']
                if std_dev: oid_match_rates_std[difficulty] = metrics['oids']['perfect_match_rate_std']
                if formatted_columns and 'columns_formatted' in metrics:
                    column_match_rates[difficulty] = metrics['columns_formatted']['perfect_match_rate']
                    if std_dev: column_match_rates_std[difficulty] = metrics['columns_formatted']['perfect_match_rate_std']
                else:
                    column_match_rates[difficulty] = metrics['columns']['perfect_match_rate']
                    if std_dev: column_match_rates_std[difficulty] = metrics['columns']['perfect_match_rate_std']
            # Add standard deviation to the match rates
            if std_dev:
                experiment_labels[exp_label] = {
                    'oid': oid_match_rates,
                    'oid_std': oid_match_rates_std,
                    'column': column_match_rates,
                    'column_std': column_match_rates_std
                }
            else:
                experiment_labels[exp_label] = {
                    'oid': oid_match_rates,
                    'column': column_match_rates
                }
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    # Figure with two subplots: OID and Column matches
    # Each subplot will be separated by difficulty on the x-axis
    # Will be bar plots for each difficulty by experiment
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # difficulties = list(next(iter(experiment_labels.values()))['oid'].keys())
    # order difficulties: simple, medium, advanced
    difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    x = np.arange(len(difficulties))  # the label locations
    width = 0.3  # the width of the bars
    std = []
    std_ = []
    std__ = []
    for i, (exp_label, data) in enumerate(experiment_labels.items()):
        oid_rates = [data['oid'][d] for d in difficulties]
        column_rates = [data['column'][d] for d in difficulties]
        # Plot OID matches
        axs[0].bar(x + (i - 0.5) * width, oid_rates, width, label=exp_label)
        # If std_dev is used, add error bars
        # Plot Column matches
        axs[1].bar(x + (i - 0.5) * width, column_rates, width, label=exp_label, )
        if std_dev:
            oid_std = [data['oid_std'][d] for d in difficulties]
            std_.extend(oid_std)
            axs[0].errorbar(x + (i - 0.5) * width, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5)
            column_std = [data['column_std'][d] for d in difficulties]
            std__.extend(column_std)
            axs[1].errorbar(x + (i - 0.5) * width, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5)
            # for n in range(len(difficulties)):
            #     std_.append(oid_std[n])
            #     std_.append(column_std[n])
    std.append(std_)
    std.append(std__)
    # Set x-ticks and labels
    axs[0].set_ylabel('Rows', fontsize=20)
    axs[1].set_ylabel('Columns', fontsize=20)
    
    axs[1].set_xticks(x)
    # change advanced to hard and change first letter to uppercase
    difficulties = [d.replace('advanced', 'hard').capitalize() for d in difficulties]
    axs[1].set_xticklabels(difficulties, fontsize=18)
    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0].legend(bbox_to_anchor=(1.0, 1.0))
    # set suptitle
    plt.suptitle('Query Generation Strategies', fontsize=24, y=0.92)
    # Add values on top of the bars
    for j, ax in enumerate(axs):
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # print(height)
            # Add the value on top of the bar, just above the standard bar height
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 + std[j][i], f"{height:.2f}", ha='center', va='bottom')            
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.04)
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])    
    # Set title by each column
    plt.subplots_adjust(hspace=0.0, )
    # plt.tight_layout()
    # plt.xlabel('Difficulties')
    plt.legend( bbox_to_anchor=(1.02, 1.0),)
    
    for ax in axs:
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()


def plot_perfect_match_by_difficulty_models(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_names: List[str],
        exp_names: List[str],
        std_dev: Union[str, None] = None,
        self_corr: bool = True,
        formatted_columns: bool = True,
        take_error_golds: bool = False,
        save_fig: Union[str, None] = None
) -> None:
    """
    Plots the perfect match rates for OID and column matches by difficulty from evaluation results for a specific model and experiments.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_names (List[str]): A list of experiment names to include in the plot.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float},
                            'difficulty': str  # e.g., 'easy', 'medium', 'hard'
                        }
                    }
                }
            }
        }
    """
    
    # Initialize data collection lists
    experiment_labels = {}
    difficulties = []
    # Self-corrected results are expected
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'

    from llm.utils.eval_utils import metrics_aggregation

    # Collect data for plotting
    models = sorted(model_names)
    for model in models:
        experiments = sorted([exp for exp in evaluation_results[model].keys() if exp in exp_names])
        for experiment in experiments:
            try:
                if "dir" in experiment:
                    exp_label = "Direct"
                if "sbs" in experiment:
                    exp_label = "Step-by-Step"

                oid_match_rates = {}
                column_match_rates = {}
                if std_dev:
                    oid_match_rates_std = {}
                    column_match_rates_std = {}

                results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                aggregate_metrics = metrics_aggregation(results=results, take_error_gold=take_error_golds)
                
                print(f"================================ {model} - {experiment} ================================")
                print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                print(f"Gold queries with errors (IDs): {np.unique(aggregate_metrics['errors']['gold_list'])}")
                # Use the aggregate metrics directly
                if std_dev == 'all' or std_dev is None:
                    # results = evaluation_results[model][experiment][self_corr_key]['aggregate_metrics']['by_difficulty']
                    results = aggregate_metrics['by_difficulty']
                elif std_dev == 'run':
                    results = aggregate_metrics['by_difficulty_runs']
                elif std_dev == 'req_id':
                    results = aggregate_metrics['by_difficulty_req_id']
                else:
                    raise ValueError(f"Unknown standard deviation type: {std_dev}")
                # iterate through difficulties
                for difficulty, metrics in results.items():
                    if difficulty not in difficulties:
                        difficulties.append(difficulty)
                    oid_match_rates[difficulty] = metrics['oids']['perfect_match_rate']
                    if std_dev: oid_match_rates_std[difficulty] = metrics['oids']['perfect_match_rate_std']
                    if formatted_columns and 'columns_formatted' in metrics:
                        column_match_rates[difficulty] = metrics['columns_formatted']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns_formatted']['perfect_match_rate_std']
                    else:
                        column_match_rates[difficulty] = metrics['columns']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns']['perfect_match_rate_std']
                # Add standard deviation to the match rates
                if std_dev:
                    experiment_labels[model] = {
                        'oid': oid_match_rates,
                        'oid_std': oid_match_rates_std,
                        'column': column_match_rates,
                        'column_std': column_match_rates_std
                    }
                else:
                    experiment_labels[model] = {
                        'oid': oid_match_rates,
                        'column': column_match_rates
                    }
            except KeyError as e:
                print(f"Warning: Missing data for {model} - {experiment}: {e}")
    # Figure with two subplots: OID and Column matches
    # Each subplot will be separated by difficulty on the x-axis
    # Will be bar plots for each difficulty by experiment
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # difficulties = list(next(iter(experiment_labels.values()))['oid'].keys())
    # order difficulties: simple, medium, advanced
    difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    x = np.arange(len(difficulties))  # the label locations
    width = 0.1  # the width of the bars
    std = []
    std_ = []
    std__ = []
    for i, (exp_name, data) in enumerate(experiment_labels.items()):
        oid_rates = [data['oid'][d] for d in difficulties]
        column_rates = [data['column'][d] for d in difficulties]
        # Plot OID matches
        if 'claude-3-7' in exp_name: exp_label = 'Claude-3.7'
        elif 'gpt' in exp_name: 
            exp_label = "-".join(exp_name.split('-')[0:2]).replace('gpt', 'GPT')
        else: exp_label = exp_name
        axs[0].bar(x + (i - 1.5) * width, oid_rates, width, label=exp_label, align='center')
        # If std_dev is used, add error bars
        # Plot Column matches
        axs[1].bar(x + (i - 1.5) * width, column_rates, width, label=exp_label, align='center')
        if std_dev:
            oid_std = [data['oid_std'][d] for d in difficulties]
            std_.extend(oid_std)
            axs[0].errorbar(x + (i - 1.5) * width, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5, elinewidth=1)
            column_std = [data['column_std'][d] for d in difficulties]
            std__.extend(column_std)
            axs[1].errorbar(x + (i - 1.5) * width, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5, elinewidth=1)
            # for n in range(len(difficulties)):
            #     std_.append(oid_std[n])
            #     std_.append(column_std[n])
    std.append(std_)
    std.append(std__)
    # Set x-ticks and labels
    axs[0].set_ylabel('Rows', fontsize=20)
    axs[1].set_ylabel('Columns', fontsize=20)
    
    # change advanced to hard and change first letter to uppercase
    difficulties = [d.replace('advanced', 'hard').capitalize() for d in difficulties]
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(difficulties, fontsize=18)
    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0].legend(bbox_to_anchor=(1.005, 1.02))
    # set suptitle
    plt.suptitle('LLM Comparison', fontsize=24, y=0.95)
    # Add values on top of the bars
    for j, ax in enumerate(axs):
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # Add the value on top of the bar, just above the standard bar height
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 + std[j][i], f"{height:.2f}", ha='center', va='bottom')            
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.04)
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])    
    # Set title by each column
    plt.subplots_adjust(hspace=0.0, )
    # plt.tight_layout()
    # plt.xlabel('Difficulties')
    # plt.legend( bbox_to_anchor=(1.02, 1.0),)
    
    for ax in axs:
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()



def table_perfect_match_by_difficulty_models(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    std_dev: Union[str, None] = None,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates separate tables showing perfect match rates for OID and column matches by difficulty and self-correction status.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                            for models and experiments, and values containing metrics.
        model_names (List[str]): List of model names to include in the table.
        exp_names (List[str]): A list of experiment names to include in the table.
        std_dev (str): The standard deviation type to use for the table. Options are 'run', 'req_id', or 'all'.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.
        take_error_golds (bool): Whether to include gold queries with errors.
        table_size (str): Size of the table. Options: 'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large'.
        adjust_margins (bool): If True, adds margin adjustment commands to fit table in page.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two tables - one for rows and one for columns.
    """
    
    from llm.utils.eval_utils import metrics_aggregation
    
    # Initialize data collection
    rows_data = {}
    columns_data = {}
    difficulties = set()
    
    # Collect data for both corrected and self_corrected
    for self_corr in [False, True]:
        self_corr_key = 'self_corrected' if self_corr else 'corrected'
        self_corr_label = 'W/o Self-Correction' if not self_corr else 'W/ Self-Correction'
        
        for model in sorted(model_names):
            if model not in rows_data:
                rows_data[model] = {}
                columns_data[model] = {}
            
            experiments = sorted([exp for exp in evaluation_results[model].keys() if exp in exp_names])
            
            # Aggregate metrics across all experiments for this model
            all_results = []
            for experiment in experiments:
                try:
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    all_results.extend(results)
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment} - {self_corr_key}: {e}")
                    continue
            
            if not all_results:
                continue
            
            # Aggregate metrics across all experiments
            aggregate_metrics = metrics_aggregation(results=all_results, take_error_gold=take_error_golds)
            
            print(f"================================ {model} - {self_corr_label} ================================")
            print(f"Number of gold queries with errors: {aggregate_metrics['errors']['gold_errors']}")
            
            # Use the appropriate metrics based on std_dev setting
            if std_dev == 'all' or std_dev is None:
                results_by_diff = aggregate_metrics['by_difficulty']
            elif std_dev == 'run':
                results_by_diff = aggregate_metrics['by_difficulty_runs']
            elif std_dev == 'req_id':
                results_by_diff = aggregate_metrics['by_difficulty_req_id']
            else:
                raise ValueError(f"Unknown standard deviation type: {std_dev}")
            
            # Extract metrics for each difficulty
            for difficulty, metrics in results_by_diff.items():
                difficulties.add(difficulty)
                
                # OID metrics
                oid_mean = metrics['oids']['perfect_match_rate']
                oid_std = metrics['oids'].get('perfect_match_rate_std', 0) if std_dev else 0
                
                # Column metrics
                if formatted_columns and 'columns_formatted' in metrics:
                    col_mean = metrics['columns_formatted']['perfect_match_rate']
                    col_std = metrics['columns_formatted'].get('perfect_match_rate_std', 0) if std_dev else 0
                else:
                    col_mean = metrics['columns']['perfect_match_rate']
                    col_std = metrics['columns'].get('perfect_match_rate_std', 0) if std_dev else 0
                
                # Store in separate table data
                difficulty_clean = difficulty.replace('advanced', 'hard')
                col_key = (self_corr_label, difficulty_clean.capitalize())
                
                if std_dev:
                    rows_data[model][col_key] = f"{oid_mean:.3f} ± {oid_std:.3f}"
                    columns_data[model][col_key] = f"{col_mean:.3f} ± {col_std:.3f}"
                else:
                    rows_data[model][col_key] = f"{oid_mean:.3f}"
                    columns_data[model][col_key] = f"{col_mean:.3f}"
    
    # Sort difficulties
    difficulties = sorted(list(difficulties), key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    
    # Create MultiIndex columns
    columns_index = []
    for self_corr_label in ['W/o Self-Correction', 'W/ Self-Correction']:
        for difficulty in difficulties:
            difficulty_clean = difficulty.replace('advanced', 'hard')
            columns_index.append((self_corr_label, difficulty_clean.capitalize()))
    
    # Create DataFrames with MultiIndex columns
    rows_df = pd.DataFrame.from_dict(rows_data, orient='index')
    columns_df = pd.DataFrame.from_dict(columns_data, orient='index')
    
    # Reorder columns to match the desired structure
    rows_df = rows_df.reindex(columns=columns_index)
    columns_df = columns_df.reindex(columns=columns_index)
    
    # Create MultiIndex for columns
    multi_columns = pd.MultiIndex.from_tuples(columns_index, names=['Self-Correction', 'Difficulty'])
    rows_df.columns = multi_columns
    columns_df.columns = multi_columns
    
    # Fill NaN values with empty string
    rows_df = rows_df.fillna('')
    columns_df = columns_df.fillna('')
    
    # Find best results per column for bold formatting
    def format_best_values(df):
        """Add bold formatting to the best (highest) values in each column"""
        df_formatted = df.copy()
        
        for col in df.columns:
            # Extract numeric values from strings like "0.123 ± 0.045" or "0.123"
            numeric_values = []
            for val in df[col]:
                if val == '':
                    numeric_values.append(-1)  # Use -1 for empty values
                else:
                    # Extract the first number (before ±)
                    try:
                        num = float(val.split(' ±')[0]) if ' ±' in val else float(val)
                        numeric_values.append(num)
                    except ValueError:
                        numeric_values.append(-1)
            
            # Find the maximum value and its indices
            max_val = max(numeric_values)
            if max_val > 0:  # Only format if there are valid values
                max_indices = [i for i, val in enumerate(numeric_values) if val == max_val]
                
                # Add bold formatting to the best values
                for idx in max_indices:
                    original_val = df_formatted.iloc[idx, df_formatted.columns.get_loc(col)]
                    if original_val != '':
                        df_formatted.iloc[idx, df_formatted.columns.get_loc(col)] = f"\\textbf{{{original_val}}}"
        
        return df_formatted
    
    def create_enhanced_latex_table(df_formatted, caption, label, table_size='small', adjust_margins=False):
        """Create enhanced LaTeX table with proper formatting and optional margin adjustment"""
        # Determine number of difficulties (assumes equal columns for each self-correction method)
        n_difficulties = len(df_formatted.columns) // 2
        
        # Create column format with vertical line between W/o and W/ self-correction (no line at the end)
        col_format = '|c|' + 'c' * n_difficulties + '|' + 'c' * n_difficulties + '|'
        
        # Generate LaTeX table
        latex_str = df_formatted.to_latex(
            escape=False, 
            multicolumn_format='c',
            column_format=col_format,
            caption=caption,
            label=label,
            position='htbp'
        )
        
        # Add table size and centering
        size_commands = {
            'tiny': '\\tiny',
            'scriptsize': '\\scriptsize', 
            'footnotesize': '\\footnotesize',
            'small': '\\small',
            'normalsize': '\\normalsize',
            'large': '\\large'
        }
        
        size_cmd = size_commands.get(table_size, '\\small')
        
        # Prepare margin adjustment commands
        margin_start = ""
        margin_end = ""
        if adjust_margins:
            margin_start = """\\adjustbox{width=\\textwidth,center}{
"""
            margin_end = """
}"""
        
        # Replace formatting
        latex_str = latex_str.replace('\\begin{tabular}', f'{size_cmd}\n\\centering\n{margin_start}\\begin{{tabular}}')
        latex_str = latex_str.replace('\\end{tabular}', f'\\end{{tabular}}{margin_end}')
        latex_str = latex_str.replace('\\toprule', '\\hline')
        latex_str = latex_str.replace('\\bottomrule', '\\hline')
        
        # Process lines to add hline between self-correction and difficulty rows
        lines = latex_str.split('\n')
        new_lines = []
        header_count = 0
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            # Count header rows and add hline between first and second header row
            if ('W/o Self-Correction' in line and 'W/ Self-Correction' in line) or \
               any(difficulty in line for difficulty in ['Simple', 'Medium', 'Hard']):
                header_count += 1
                # Add hline after first header row (self-correction row)
                if header_count == 1:
                    new_lines.append('\\hline')
        
        # Remove midrule replacements
        latex_str_modified = '\n'.join(new_lines)
        latex_str_modified = latex_str_modified.replace('\\midrule', '')
        
        # Add adjustbox package note if margins are adjusted
        if adjust_margins:
            package_note = "% Note: Add \\usepackage{adjustbox} to your LaTeX preamble for margin adjustment\n"
            latex_str_modified = package_note + latex_str_modified
        
        return latex_str_modified
    
    # Format best values with bold
    rows_df_formatted = format_best_values(rows_df)
    columns_df_formatted = format_best_values(columns_df)
    
    # Print tables in LaTeX format with enhanced formatting
    print("\n" + "="*80)
    print("ROWS (Perfect Match Rates) - LaTeX Format")
    print("="*80)

    latex_str_rows = create_enhanced_latex_table(
        rows_df_formatted, 
        'Perfect Match Rates for Rows',
        'tab:rows_perfect_match',
        table_size,
        adjust_margins
    )
    print(latex_str_rows)

    print("\n" + "="*80)
    print("COLUMNS (Perfect Match Rates) - LaTeX Format") 
    print("="*80)

    latex_str_columns = create_enhanced_latex_table(
        columns_df_formatted,
        'Perfect Match Rates for Columns', 
        'tab:columns_perfect_match',
        table_size,
        adjust_margins
    )
    print(latex_str_columns)

    print("\n" + "="*80)
    print("Available table sizes: tiny, scriptsize, footnotesize, small, normalsize, large")
    print("Current table size:", table_size)
    print("Margin adjustment:", "enabled" if adjust_margins else "disabled")
    print("="*80)
    
    return rows_df_formatted, columns_df_formatted



import pandas as pd
from typing import Dict, List, Union, Any, Tuple

# Helper function to extract numeric value for comparison
def _get_value(val_str: Union[str, None]) -> float:
    if val_str is None:
        return -1.0
    try:
        # Extract number before " $\pm$" or just the number
        clean = val_str.split(' $')[0].split(' ±')[0]
        return float(clean)
    except:
        return -1.0

def generate_latex_table_by_experiment(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    std_dev: Union[str, None] = None,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
) -> str:
    """
    Creates a SINGLE table with a SHARED header, split vertically by 
    Metric Type (OID/Column) AND by experiment name.
    
    This stacks all results into one large table, e.g.:
    - OID Match - Experiment 1
    - OID Match - Experiment 2
    - Column Match - Experiment 1
    - Column Match - Experiment 2
    
    Args:
        evaluation_results: Nested dict of evaluation results.
        model_names: List of models to include.
        exp_names: List of experiment names to include *separately*.
        std_dev: Standard deviation type ('run', 'req_id', 'all', or None).
        formatted_columns: Whether to use 'columns_formatted' metrics.
        take_error_golds: Whether to include gold queries with errors.
        table_size: LaTeX font size command.
        adjust_margins: Whether to use 'table*' and squeeze columns.

    Returns:
        str: A string containing the complete, ready-to-use LaTeX table.
    """
    
    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
        print("Please ensure this utility is available in your environment.")
        # Define a basic placeholder if it's not found, to avoid crashing
        # This assumes a very specific structure and is NOT a real replacement.
        def metrics_aggregation(results, take_error_gold):
            print("Using placeholder 'metrics_aggregation' due to import error.")
            # This is a fallback and will likely fail if data is complex.
            # A more robust solution would be needed if this is a common issue.
            if results:
                return results[0].get('summary', {})
            return {}

    
    # --- 1. Data Collection ---
    # We store data in a flat map for easy lookup
    # Key: (Metric_Type, Exp_Name, Model, SC_Status, Difficulty)
    data_map = {} 
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    for exp_name in sorted_exp_names:
        for model in sorted_models:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue # Skip if model or experiment data is missing
            
            for self_corr in [False, True]:
                self_corr_key = 'self_corrected' if self_corr else 'corrected'
                sc_label = 'W/ Self-Corr' if self_corr else 'W/o Self-Corr'
                
                try:
                    # Get results for this *specific* experiment
                    results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
                except KeyError:
                    # This experiment/model/sc_key combo might not exist
                    continue
                
                if not results_list:
                    continue
                    
                # Aggregate metrics for *this experiment's results only*
                agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
                
                if std_dev == 'run':
                    results_by_diff = agg_metrics.get('by_difficulty_runs', {})
                elif std_dev == 'req_id':
                    results_by_diff = agg_metrics.get('by_difficulty_req_id', {})
                else:
                    results_by_diff = agg_metrics.get('by_difficulty', {}) # Default or 'all'

                for diff, metrics in results_by_diff.items():
                    diff_clean = diff.replace('advanced', 'hard').capitalize()
                    difficulties_found.add(diff_clean)
                    
                    # --- ROW (OID) METRICS ---
                    oid_mean = metrics.get('oids', {}).get('perfect_match_rate', 0)
                    oid_std = metrics.get('oids', {}).get('perfect_match_rate_std', 0) if std_dev else 0
                    
                    # --- COLUMN METRICS ---
                    col_metrics = metrics.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
                    col_mean = col_metrics.get('perfect_match_rate', 0)
                    col_std = col_metrics.get('perfect_match_rate_std', 0) if std_dev else 0

                    # Format Strings
                    val_fmt = "{:.3f}"
                    if std_dev:
                        val_fmt += " $\pm$ {:.3f}"
                        row_str = val_fmt.format(oid_mean, oid_std)
                        col_str = val_fmt.format(col_mean, col_std)
                    else:
                        row_str = val_fmt.format(oid_mean)
                        col_str = val_fmt.format(col_mean)
                    
                    data_map[('ID Match', exp_name, model, sc_label, diff_clean)] = row_str
                    data_map[('Column Match', exp_name, model, sc_label, diff_clean)] = col_str

    # --- 2. Find Max Values for Bolding (within each metric + experiment group) ---
    
    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)
    sc_labels = ['W/o Self-Corr', 'W/ Self-Corr']
    metric_labels = ['ID Match', 'Column Match']
    
    max_vals = {} # Key: (Metric, Exp_Name, Difficulty, SC_Status) -> max_value
    for metric in metric_labels:
        for exp_name in sorted_exp_names:
            for diff in sorted_diffs:
                for sc in sc_labels:
                    vals = [_get_value(data_map.get((metric, exp_name, model, sc, diff))) for model in sorted_models]
                    max_vals[(metric, exp_name, diff, sc)] = max(vals) if (vals and any(v > -1 for v in vals)) else -1

    # --- 3. Manually Build LaTeX Table String ---
    
    latex_lines = []
    
    # Calculate column counts
    num_data_cols = len(sorted_diffs) * 2
    total_cols = 1 + num_data_cols # 1 for Model stub
    
    # Preamble
    table_env = "table*" if adjust_margins else "table"
    latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
    
    size_commands = {
        'tiny': '\\tiny', 'scriptsize': '\\scriptsize', 'footnotesize': '\\footnotesize',
        'small': '\\small', 'normalsize': '\\normalsize', 'large': '\\large'
    }
    latex_lines.append(size_commands.get(table_size, '\\small'))
    
    latex_lines.append(r"\centering")
    
    if adjust_margins:
        latex_lines.append(r"\setlength{\tabcolsep}{4pt}") 
    
    col_def = "l" + ("c" * num_data_cols)
    latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex_lines.append(r"\toprule")

    # -- Header Row 1: Difficulties (Spanned) --
    header_1 = [""] # Empty for stub
    for diff in sorted_diffs:
        header_1.append(f"\\multicolumn{{2}}{{c}}{{{diff}}}")
    latex_lines.append(" & ".join(header_1) + r" \\")

    # -- Cmidrules for Difficulty Spans --
    cmid_rules = []
    for i in range(len(sorted_diffs)):
        start_col = 2 + (i * 2)
        end_col = 3 + (i * 2)
        cmid_rules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    latex_lines.append(" ".join(cmid_rules))
    
    # -- Header Row 2: Self-Correction (Decked) --
    header_2 = ["Model"] # Stub header
    for _ in sorted_diffs:
        header_2.append("W/o Self-Corr")
        header_2.append("W/ Self-Corr")
    latex_lines.append(" & ".join(header_2) + r" \\")
    latex_lines.append(r"\midrule")

    # -- Body: Iterate through metrics, experiments, and models --
    for i, metric in enumerate(metric_labels):
        if i > 0:
            latex_lines.append(r"\midrule") # Separator between OID and Column
            
        for j, exp_name in enumerate(sorted_exp_names):
            if j > 0:
                # Add a lighter rule between experiments *within* the same metric block
                latex_lines.append(f"\\cmidrule(lr){{1-{total_cols}}}")
            
            # -- Row Spanner for Metric Type + Experiment --
            metric_label = "ID Match (Rows)" if metric == "ID Match" else "Column Match"
            # Clean up exp_name for display (e.g., remove underscores)
            # exp_display = exp_name.replace("_", " ").title()
            exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
            exp_display = 'Direct' if 'direct' in exp_name else exp_display
            spanner_label = f"\\textbf{{{metric_label} -- {exp_display}}}"
            latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{{spanner_label}}} \\\\")
            
            # -- Data Rows for this Metric + Experiment --
            for model in sorted_models:
                if 'claude-3-7' in model: model_label = 'Claude-3.7'
                elif 'gpt' in model: 
                    model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
                else: model_label = model
                line = [model_label] # Start with model name
                has_data = False
                for diff in sorted_diffs:
                    for sc in sc_labels:
                        val_str = data_map.get((metric, exp_name, model, sc, diff), "-")
                        if val_str != "-":
                            has_data = True
                        val_num = _get_value(val_str)
                        max_val = max_vals.get((metric, exp_name, diff, sc), -1)
                        
                        # Apply bold if it's the max for this group
                        if abs(val_num - max_val) < 1e-6 and max_val >= 0:
                            line.append(f"\\textbf{{{val_str}}}")
                        else:
                            line.append(val_str)
                
                # Only add the row if it contains data
                if has_data:
                    latex_lines.append(" & ".join(line) + r" \\")

    # -- Footer --
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    # Caption and Label
    latex_lines.append(r"\caption{Perfect Match Rates by Metric, Experiment, Difficulty, and Self-Correction. Best result in each group and column is bolded.}")
    latex_lines.append(r"\label{tab:combined_vertical_results_by_exp}")
    
    latex_lines.append(f"\\end{{{table_env}}}")
    
    final_latex_string = "\n".join(latex_lines)
    
    # Print for user
    print("\n" + "="*80)
    print("Vertically Split LaTeX Table by Experiment (Booktabs Format)")
    print("="*80)
    print(final_latex_string)
    print("="*80)
    
    return final_latex_string


# def generate_latex_tables_by_sc_status(
#     evaluation_results: Dict[str, Dict[str, Any]],
#     model_names: List[str],
#     exp_names: List[str],
#     std_dev: Union[str, None] = None,
#     formatted_columns: bool = True,
#     take_error_golds: bool = False,
#     table_size: str = 'small',
#     adjust_margins: bool = True,
# ) -> str:
#     """
#     Creates TWO SEPARATE tables, one for "W/o Self-Correction" and one 
#     for "W/ Self-Correction".
    
#     Each table maintains the vertical stacking format (i.e., stacked by 
#     Metric Type and Experiment Name).
    
#     The headers are simplified to only show difficulties, as the
#     self-correction status is defined by the table itself.

#     Args:
#         [Same as generate_latex_table_by_experiment]

#     Returns:
#         str: A single string containing the LaTeX for BOTH tables,
#              one after the other, separated by a '\\bigskip'.
#     """
    
#     # Attempt to import the aggregation utility
#     try:
#         from llm.utils.eval_utils import metrics_aggregation
#     except ImportError:
#         print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
#         print("Please ensure this utility is available in your environment.")
#         def metrics_aggregation(results, take_error_gold):
#             print("Using placeholder 'metrics_aggregation' due to import error.")
#             if results:
#                 # This is a dangerous assumption, update if necessary
#                 return results[0].get('summary', {})
#             return {}

    
#     # --- 1. Data Collection ---
#     # This logic is identical to generate_latex_table_by_experiment
#     # We collect all data first.
#     # Key: (Metric_Type, Exp_Name, Model, SC_Status, Difficulty)
#     data_map = {} 
#     difficulties_found = set()
#     sorted_exp_names = sorted(exp_names)
#     sorted_models = sorted(model_names)
    
#     for exp_name in sorted_exp_names:
#         for model in sorted_models:
#             if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
#                 continue 
            
#             for self_corr in [False, True]:
#                 self_corr_key = 'self_corrected' if self_corr else 'corrected'
#                 sc_label = 'W/ Self-Corr' if self_corr else 'W/o Self-Corr'
                
#                 try:
#                     results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
#                 except KeyError:
#                     continue
                
#                 if not results_list:
#                     continue
                    
#                 agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
                
#                 if std_dev == 'run':
#                     results_by_diff = agg_metrics.get('by_difficulty_runs', {})
#                 elif std_dev == 'req_id':
#                     results_by_diff = agg_metrics.get('by_difficulty_req_id', {})
#                 else:
#                     results_by_diff = agg_metrics.get('by_difficulty', {}) 

#                 for diff, metrics in results_by_diff.items():
#                     diff_clean = diff.replace('advanced', 'hard').capitalize()
#                     difficulties_found.add(diff_clean)
                    
#                     oid_mean = metrics.get('oids', {}).get('perfect_match_rate', 0)
#                     oid_std = metrics.get('oids', {}).get('perfect_match_rate_std', 0) if std_dev else 0
                    
#                     col_metrics = metrics.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
#                     col_mean = col_metrics.get('perfect_match_rate', 0)
#                     col_std = col_metrics.get('perfect_match_rate_std', 0) if std_dev else 0

#                     val_fmt = "{:.3f}"
#                     if std_dev:
#                         val_fmt += " $\pm$ {:.3f}"
#                         row_str = val_fmt.format(oid_mean, oid_std)
#                         col_str = val_fmt.format(col_mean, col_std)
#                     else:
#                         row_str = val_fmt.format(oid_mean)
#                         col_str = val_fmt.format(col_mean)

#                     data_map[('ID Match', exp_name, model, sc_label, diff_clean)] = row_str
#                     data_map[('Column Match', exp_name, model, sc_label, diff_clean)] = col_str

#     # --- 2. Find Max Values for Bolding (within each metric + experiment group) ---
#     # This logic is also identical, as we need all max values pre-calculated.
#     diff_order = ['Simple', 'Medium', 'Hard']
#     sorted_diffs = sorted(list(difficulties_found), 
#                           key=lambda x: diff_order.index(x) if x in diff_order else 99)
#     sc_labels = ['W/o Self-Corr', 'W/ Self-Corr']
#     metric_labels = ['ID Match', 'Column Match']
    
#     max_vals = {} # Key: (Metric, Exp_Name, Difficulty, SC_Status) -> max_value
#     for metric in metric_labels:
#         for exp_name in sorted_exp_names:
#             for diff in sorted_diffs:
#                 for sc in sc_labels:
#                     vals = [_get_value(data_map.get((metric, exp_name, model, sc, diff))) for model in sorted_models]
#                     max_vals[(metric, exp_name, diff, sc)] = max(vals) if (vals and any(v > -1 for v in vals)) else -1

#     # --- 3. Manually Build LaTeX Table Strings (Looping for each SC status) ---
    
#     all_latex_tables = []
    
#     for sc_bool, sc_title in [(False, "W/o Self-Correction"), (True, "W/ Self-Correction")]:
        
#         # This is the key to get data from our map
#         sc_label_key = "W/o Self-Corr" if not sc_bool else "W/ Self-Corr"
        
#         latex_lines = []
        
#         # Calculate column counts (Simplified header)
#         num_data_cols = len(sorted_diffs)
#         total_cols = 1 + num_data_cols # 1 for Model stub
        
#         # Preamble
#         table_env = "table*" if adjust_margins else "table"
#         latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
        
#         size_commands = {
#             'tiny': '\\tiny', 'scriptsize': '\\scriptsize', 'footnotesize': '\\footnotesize',
#             'small': '\\small', 'normalsize': '\\normalsize', 'large': '\\large'
#         }
#         latex_lines.append(size_commands.get(table_size, '\\small'))
        
#         latex_lines.append(r"\centering")
        
#         if adjust_margins:
#             latex_lines.append(r"\setlength{\tabcolsep}{4pt}") 
        
#         # Simplified column definition
#         col_def = "l" + ("c" * num_data_cols)
#         latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
#         latex_lines.append(r"\toprule")

#         # -- Simplified Header Row --
#         header = ["Model"] + sorted_diffs
#         latex_lines.append(" & ".join(header) + r" \\")
#         latex_lines.append(r"\midrule")

#         # -- Body: Iterate through metrics, experiments, and models --
#         for i, metric in enumerate(metric_labels):
#             if i > 0:
#                 latex_lines.append(r"\midrule") # Separator between OID and Column
                
#             for j, exp_name in enumerate(sorted_exp_names):
#                 if j > 0:
#                     latex_lines.append(f"\\cmidrule(lr){{1-{total_cols}}}")
                
#                 # -- Row Spanner for Metric Type + Experiment --
#                 metric_label = "ID Match (Rows)" if metric == "ID Match" else "Column Match"
#                 # exp_display = exp_name.replace("_", " ").title()
#                 exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
#                 exp_display = 'Direct' if 'direct' in exp_name else exp_display
#                 spanner_label = f"\\textbf{{{metric_label} -- {exp_display}}}"
#                 latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{{spanner_label}}} \\\\")
                
#                 # -- Data Rows for this Metric + Experiment --
#                 for model in sorted_models:
#                     if 'claude-3-7' in model: model_label = 'Claude-3.7'
#                     elif 'gpt' in model: 
#                         model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
#                     else: model_label = model
#                     line = [model_label] # Start with model name
#                     has_data = False
#                     for diff in sorted_diffs:
#                         # *** KEY CHANGE ***
#                         # Get data for the specific SC status of this table
#                         val_str = data_map.get((metric, exp_name, model, sc_label_key, diff), "-")
                        
#                         if val_str != "-":
#                             has_data = True
#                         val_num = _get_value(val_str)
                        
#                         # Get max val for this specific SC status
#                         max_val = max_vals.get((metric, exp_name, diff, sc_label_key), -1)
                        
#                         if abs(val_num - max_val) < 1e-6 and max_val >= 0:
#                             line.append(f"\\textbf{{{val_str}}}")
#                         else:
#                             line.append(val_str)
                    
#                     if has_data:
#                         latex_lines.append(" & ".join(line) + r" \\")

#         # -- Footer --
#         latex_lines.append(r"\bottomrule")
#         latex_lines.append(r"\end{tabular}")
        
#         # -- Caption and Label (Customized for SC status) --
#         caption = (
#             "Perfect Match Rates by Metric and Experiment "
#             f"({sc_title}). Best result in each group "
#             "and column is bolded."
#         )
#         latex_lines.append(f"\\caption{{{caption}}}")
        
#         label_sc_suffix = "_wo_sc" if not sc_bool else "_w_sc"
#         latex_lines.append(f"\\label{{tab:vertical_by_exp{label_sc_suffix}}}")
        
#         latex_lines.append(f"\\end{{{table_env}}}")
        
#         all_latex_tables.append("\n".join(latex_lines))
    
#     # Join the two tables with a vertical space
#     final_latex_string = "\n\n\\bigskip\n\n".join(all_latex_tables)
    
#     # Print for user
#     print("\n" + "="*80)
#     print("TWO SEPARATE LaTeX Tables by Self-Correction (Booktabs Format)")
#     print("="*80)
#     print(final_latex_string)
#     print("="*80)
    
#     return final_latex_string


def generate_latex_tables_by_sc_status(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    std_dev: Union[str, None] = None,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
    parallel_metrics: bool = False, # <-- NEW INPUT
) -> str:
    """
    Creates TWO SEPARATE tables, one for "W/o Self-Correction" and one 
    for "W/ Self-Correction".
    
    Can generate tables in two formats based on `parallel_metrics`:
    
    1. parallel_metrics=False (Default):
       - Long format.
       - Stacks metrics vertically (ID Match - Exp1, ID Match - Exp2, ...).
       - Headers are just 'Simple', 'Medium', 'Hard'.
       
    2. parallel_metrics=True:
       - Wide format.
       - Metrics (ID Match, Column Match) are parallel column groups.
       - Experiments (Exp1, Exp2) are vertical row spanners.
       - Headers are Metric -> Difficulty.

    Args:
        [Same as generate_latex_table_by_experiment]
        parallel_metrics (bool): Selects the table format.

    Returns:
        str: A single string containing the LaTeX for BOTH tables,
             one after the other, separated by a '\\bigskip'.
    """
    
    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
        print("Please ensure this utility is available in your environment.")
        def metrics_aggregation(results, take_error_gold):
            print("Using placeholder 'metrics_aggregation' due to import error.")
            if results:
                # This is a dangerous assumption, update if necessary
                return results[0].get('summary', {})
            return {}

    
    # --- 1. Data Collection ---
    # This logic is identical for both formats
    data_map = {} 
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    for exp_name in sorted_exp_names:
        for model in sorted_models:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue 
            
            for self_corr in [False, True]:
                self_corr_key = 'self_corrected' if self_corr else 'corrected'
                sc_label = 'W/ Self-Corr' if self_corr else 'W/o Self-Corr'
                
                try:
                    results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
                except KeyError:
                    continue
                
                if not results_list:
                    continue
                    
                agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
                
                if std_dev == 'run':
                    results_by_diff = agg_metrics.get('by_difficulty_runs', {})
                elif std_dev == 'req_id':
                    results_by_diff = agg_metrics.get('by_difficulty_req_id', {})
                else:
                    results_by_diff = agg_metrics.get('by_difficulty', {}) 

                for diff, metrics in results_by_diff.items():
                    diff_clean = diff.replace('advanced', 'hard').capitalize()
                    difficulties_found.add(diff_clean)
                    
                    oid_mean = metrics.get('oids', {}).get('perfect_match_rate', 0)
                    oid_std = metrics.get('oids', {}).get('perfect_match_rate_std', 0) if std_dev else 0
                    
                    col_metrics = metrics.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
                    col_mean = col_metrics.get('perfect_match_rate', 0)
                    col_std = col_metrics.get('perfect_match_rate_std', 0) if std_dev else 0

                    val_fmt = "{:.3f}"
                    if std_dev:
                        val_fmt += " $\pm$ {:.3f}"
                        row_str = val_fmt.format(oid_mean, oid_std)
                        col_str = val_fmt.format(col_mean, col_std)
                    else:
                        row_str = val_fmt.format(oid_mean)
                        col_str = val_fmt.format(col_mean)

                    data_map[('ID Match', exp_name, model, sc_label, diff_clean)] = row_str
                    data_map[('Column Match', exp_name, model, sc_label, diff_clean)] = col_str

    # --- 2. Find Max Values for Bolding ---
    # This logic is also identical for both formats
    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)
    sc_labels = ['W/o Self-Corr', 'W/ Self-Corr']
    metric_labels = ['ID Match', 'Column Match']
    
    max_vals = {} # Key: (Metric, Exp_Name, Difficulty, SC_Status) -> max_value
    for metric in metric_labels:
        for exp_name in sorted_exp_names:
            for diff in sorted_diffs:
                for sc in sc_labels:
                    vals = [_get_value(data_map.get((metric, exp_name, model, sc, diff))) for model in sorted_models]
                    max_vals[(metric, exp_name, diff, sc)] = max(vals) if (vals and any(v > -1 for v in vals)) else -1

    # --- 3. Manually Build LaTeX Table Strings (Looping for each SC status) ---
    
    all_latex_tables = []
    size_commands = {
        'tiny': '\\tiny', 'scriptsize': '\\scriptsize', 'footnotesize': '\\footnotesize',
        'small': '\\small', 'normalsize': '\\normalsize', 'large': '\\large'
    }
    for sc_bool, sc_title in [(False, "W/o Self-Correction"), (True, "W/ Self-Correction")]:
        
        sc_label_key = "W/o Self-Corr" if not sc_bool else "W/ Self-Corr"
        latex_lines = []
        
        table_env = "table*" if adjust_margins else "table"
        size_cmd = size_commands.get(table_size, '\\small')
        
        # --- PREAMBLE ---
        latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
        latex_lines.append(size_cmd)
        latex_lines.append(r"\centering")
        if adjust_margins:
            latex_lines.append(r"\setlength{\tabcolsep}{4pt}") 
        
        if parallel_metrics:
            # --- NEW FORMAT: WIDE (Parallel Metrics) ---
            num_diffs = len(sorted_diffs)
            num_data_cols = num_diffs * 2 # ID group + Column group
            total_cols = 1 + num_data_cols
            
            # Column def with vertical lines
            col_def = "l | " + ("c" * num_diffs) + " | " + ("c" * num_diffs)
            latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
            latex_lines.append(r"\toprule")

            # -- Header 1: Metric Type (Spanned) --
            header_1 = [""] # Empty for stub
            header_1.append(f"\\multicolumn{{{num_diffs}}}{{c|}}{{ID Match (Rows)}}")
            header_1.append(f"\\multicolumn{{{num_diffs}}}{{c|}}{{Column Match}}")
            latex_lines.append(" & ".join(header_1) + r" \\")

            # -- Header 2: Difficulties (Decked) --
            header_2 = ["Model"] + sorted_diffs + sorted_diffs # e.g., Model, S, M, H, S, M, H
            latex_lines.append(" & ".join(header_2) + r" \\")
            latex_lines.append(r"\midrule")

            # -- Body: Iterate through experiments (as spanners) and models --
            for j, exp_name in enumerate(sorted_exp_names):
                if j > 0:
                    latex_lines.append(f"\\cmidrule(lr){{1-{total_cols}}}")
                
                # -- Row Spanner for Experiment --
                exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
                exp_display = 'Direct' if 'direct' in exp_name else exp_display
                spanner_label = f"\\textbf{{{exp_display}}}"
                latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{{spanner_label}}} \\\\")
                
                # -- Data Rows for this Experiment --
                for model in sorted_models:
                    if 'claude-3-7' in model: model_label = 'Claude-3.7'
                    elif 'gpt' in model: 
                        model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
                    else: model_label = model
                    
                    line = [model_label]
                    has_data = False
                    
                    # --- ID Match Block ---
                    for diff in sorted_diffs:
                        val_str = data_map.get(('ID Match', exp_name, model, sc_label_key, diff), "-")
                        if val_str != "-": has_data = True
                        val_num = _get_value(val_str)
                        max_val = max_vals.get(('ID Match', exp_name, diff, sc_label_key), -1)
                        if abs(val_num - max_val) < 1e-6 and max_val >= 0:
                            line.append(f"\\textbf{{{val_str}}}")
                        else:
                            line.append(val_str)
                            
                    # --- Column Match Block ---
                    for diff in sorted_diffs:
                        val_str = data_map.get(('Column Match', exp_name, model, sc_label_key, diff), "-")
                        if val_str != "-": has_data = True
                        val_num = _get_value(val_str)
                        max_val = max_vals.get(('Column Match', exp_name, diff, sc_label_key), -1)
                        if abs(val_num - max_val) < 1e-6 and max_val >= 0:
                            line.append(f"\\textbf{{{val_str}}}")
                        else:
                            line.append(val_str)

                    if has_data:
                        latex_lines.append(" & ".join(line) + r" \\")

        else:
            # --- OLD FORMAT: LONG (Stacked Metrics) ---
            num_data_cols = len(sorted_diffs)
            total_cols = 1 + num_data_cols
            
            col_def = "l" + ("c" * num_data_cols)
            latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
            latex_lines.append(r"\toprule")

            # -- Simplified Header Row --
            header = ["Model"] + sorted_diffs
            latex_lines.append(" & ".join(header) + r" \\")
            latex_lines.append(r"\midrule")

            # -- Body: Iterate through metrics, experiments, and models --
            for i, metric in enumerate(metric_labels):
                if i > 0:
                    latex_lines.append(r"\midrule") # Separator between OID and Column
                    
                for j, exp_name in enumerate(sorted_exp_names):
                    if j > 0:
                        latex_lines.append(f"\\cmidrule(lr){{1-{total_cols}}}")
                    
                    metric_label = "ID Match (Rows)" if metric == "ID Match" else "Column Match"
                    exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
                    exp_display = 'Direct' if 'direct' in exp_name else exp_display
                    spanner_label = f"\\textbf{{{metric_label} -- {exp_display}}}"
                    latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{{spanner_label}}} \\\\")
                    
                    for model in sorted_models:
                        if 'claude-3-7' in model: model_label = 'Claude-3.7'
                        elif 'gpt' in model: 
                            model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
                        else: model_label = model
                        line = [model_label]
                        has_data = False
                        for diff in sorted_diffs:
                            val_str = data_map.get((metric, exp_name, model, sc_label_key, diff), "-")
                            if val_str != "-": has_data = True
                            val_num = _get_value(val_str)
                            max_val = max_vals.get((metric, exp_name, diff, sc_label_key), -1)
                            
                            if abs(val_num - max_val) < 1e-6 and max_val >= 0:
                                line.append(f"\\textbf{{{val_str}}}")
                            else:
                                line.append(val_str)
                        
                        if has_data:
                            latex_lines.append(" & ".join(line) + r" \\")

        # --- FOOTER (Common to both formats) ---
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        
        caption = (
            "Perfect Match Rates by Experiment "
            f"({sc_title}). Best result in each group "
            "and column is bolded."
        )
        latex_lines.append(f"\\caption{{{caption}}}")
        
        label_sc_suffix = "_wo_sc" if not sc_bool else "_w_sc"
        label_format = "_parallel" if parallel_metrics else "_stacked"
        latex_lines.append(f"\\label{{tab:vertical_by_exp{label_sc_suffix}{label_format}}}")
        
        latex_lines.append(f"\\end{{{table_env}}}")
        
        all_latex_tables.append("\n".join(latex_lines))
    
    # Join the two tables with a vertical space
    final_latex_string = "\n\n\\bigskip\n\n".join(all_latex_tables)
    
    # Print for user
    print("\n" + "="*80)
    print(f"TWO SEPARATE LaTeX Tables by Self-Correction (Format: {'Parallel' if parallel_metrics else 'Stacked'})")
    print("="*80)
    print(final_latex_string)
    print("="*80)
    
    return final_latex_string


def generate_latex_ranking_tables(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    use_self_correction: bool = True,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
) -> str:
    """
    Creates separate tables for each experiment, showing the *rank*
    of each model, as shown in the user-provided image.
    
    - Ranks are calculated per-column (higher score = rank 1).
    - A 'SUM' column is added.
    - Models are sorted by 'SUM' (lowest is best).
    - Uses a specific self-correction status as defined by `use_self_correction`.

    Args:
        evaluation_results: Nested dict of evaluation results.
        model_names: List of models to include.
        exp_names: List of experiment names to rank.
        use_self_correction: (bool) If True, uses "W/ Self-Corr" data. 
                             If False, uses "W/o Self-Corr" data.
        formatted_columns: Whether to use 'columns_formatted' metrics.
        take_error_golds: Whether to include gold queries with errors.
        table_size: LaTeX font size command.
        adjust_margins: Whether to use 'table*' and squeeze columns.

    Returns:
        str: A single string containing the LaTeX for ALL ranking tables,
             one after the other, separated by a '\\bigskip'.
    """
    
    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
        print("Please ensure this utility is available in your environment.")
        def metrics_aggregation(results, take_error_gold):
            print("Using placeholder 'metrics_aggregation' due to import error.")
            if results:
                return results[0].get('summary', {})
            return {}

    
    # --- 1. Data Collection ---
    # We only store the raw *mean* values for ranking.
    # Key: (Metric_Type, Exp_Name, Model, Difficulty)
    data_map = {} 
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    sc_label_key = "W/ Self-Corr" if use_self_correction else "W/o Self-Corr"
    sc_title = "W/ Self-Correction" if use_self_correction else "W/o Self-Correction"
    self_corr_key = 'self_corrected' if use_self_correction else 'corrected'
    
    for exp_name in sorted_exp_names:
        for model in sorted_models:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue 
                
            try:
                results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
            except KeyError:
                continue
            
            if not results_list:
                continue
                
            agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
            
            # We only need 'by_difficulty' for the mean scores
            results_by_diff = agg_metrics.get('by_difficulty', {}) 

            for diff, metrics in results_by_diff.items():
                diff_clean = diff.replace('advanced', 'hard').capitalize()
                difficulties_found.add(diff_clean)
                
                oid_mean = metrics.get('oids', {}).get('perfect_match_rate', 0)
                
                col_metrics = metrics.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
                col_mean = col_metrics.get('perfect_match_rate', 0)

                # Store the raw float values, not strings
                data_map[('ID Match', exp_name, model, diff_clean)] = oid_mean
                data_map[('Column Match', exp_name, model, diff_clean)] = col_mean

    # --- 2. Build and Rank Tables per Experiment ---
    
    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)
    metric_labels = ['ID Match', 'Column Match']
    
    # These are the columns from your image: RS, RM, RH, CS, CM, CH
    rank_col_headers = []
    for m_short, _ in [('R', 'ID Match'), ('C', 'Column Match')]:
        for d_short in [d[0] for d in sorted_diffs]: # S, M, H
            rank_col_headers.append(f"{m_short}{d_short}")

    all_latex_tables = []

    for exp_name in sorted_exp_names:
        
        # --- Create DataFrame with raw scores ---
        pd_data = []
        for model in sorted_models:
            # Get model display name
            if 'claude-3-7' in model: model_label = 'Claude-3.7'
            elif 'gpt' in model: 
                model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
            else: model_label = model
            
            row_data = {'Model': model_label}
            has_data = False
            for metric in metric_labels:
                for diff in sorted_diffs:
                    # e.g., 'R' + 'S' -> 'RS'
                    col_key = f"{'R' if metric == 'ID Match' else 'C'}{diff[0]}"
                    score = data_map.get((metric, exp_name, model, diff), None)
                    
                    if score is not None:
                        has_data = True
                    row_data[col_key] = score
            
            if has_data:
                pd_data.append(row_data)

        if not pd_data:
            continue # Skip if no data for this experiment

        df_scores = pd.DataFrame(pd_data).set_index('Model').fillna(-1)
        
        # --- Rank the DataFrame ---
        # ascending=False means higher scores get lower ranks (Rank 1)
        # method='min' handles ties (e.g., two 1st places, next is 3rd)
        df_ranks = df_scores.rank(ascending=False, method='min').astype(int)
        
        # --- Sum and Sort ---
        df_ranks['SUM'] = df_ranks.sum(axis=1)
        df_ranks = df_ranks.sort_values(by='SUM', ascending=True)

        # --- 3. Build LaTeX String for this table ---
        latex_lines = []
        table_env = "table*" if adjust_margins else "table"
        latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
        
        size_commands = {
            'tiny': '\\tiny', 'scriptsize': '\\scriptsize', 'footnotesize': '\\footnotesize',
            'small': '\\small', 'normalsize': '\\normalsize', 'large': '\\large'
        }
        latex_lines.append(size_commands.get(table_size, '\\small'))
        latex_lines.append(r"\centering")

        # Column definition: l | c c c | c c c || c
        # Vertical lines as seen in your drawing
        col_def = "l | " + " ".join(["c"] * len(sorted_diffs)) + " | " \
                  + " ".join(["c"] * len(sorted_diffs)) + " || c"
        
        latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
        latex_lines.append(r"\toprule")
        
        # Header Row
        header = ["LLM"] + rank_col_headers + ["SUM"]
        latex_lines.append(" & ".join(header) + r" \\")
        latex_lines.append(r"\midrule")
        
        # Data Rows
        for model_label, row in df_ranks.iterrows():
            row_values = [model_label] + [str(r) for r in row.values]
            latex_lines.append(" & ".join(row_values) + r" \\")
            
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        
        # Caption and Label
        exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
        exp_display = 'Direct' if 'direct' in exp_name else exp_display
        
        caption = f"Model Ranking for {exp_display} ({sc_title})"
        latex_lines.append(f"\\caption{{{caption}}}")
        
        label_exp = "_sbs" if 'sbs' in exp_name else "_direct"
        label_sc = "_w_sc" if use_self_correction else "_wo_sc"
        latex_lines.append(f"\\label{{tab:rank_{label_exp}{label_sc}}}")
        
        latex_lines.append(f"\\end{{{table_env}}}")
        all_latex_tables.append("\n".join(latex_lines))

    # Join all tables
    final_latex_string = "\n\n\\bigskip\n\n".join(all_latex_tables)
    
    # Print for user
    print("\n" + "="*80)
    print(f"LaTeX Ranking Tables ({sc_title})")
    print("="*80)
    print(final_latex_string)
    print("="*80)
    
    return final_latex_string


def generate_latex_table_metrics_by_experiment(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    std_dev: Union[str, None] = None,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
    metric_type: List[str] = 'perfect_match_rate',
) -> str:
    """
    Creates a SINGLE table with a SHARED header, split vertically by 
    Metric Type (OID/Column) AND by experiment name.
    
    This stacks all results into one large table, e.g.:
    - OID Match - Experiment 1
    - OID Match - Experiment 2
    - Column Match - Experiment 1
    - Column Match - Experiment 2
    
    Args:
        evaluation_results: Nested dict of evaluation results.
        model_names: List of models to include.
        exp_names: List of experiment names to include *separately*.
        std_dev: Standard deviation type ('run', 'req_id', 'all', or None).
        formatted_columns: Whether to use 'columns_formatted' metrics.
        take_error_golds: Whether to include gold queries with errors.
        table_size: LaTeX font size command.
        adjust_margins: Whether to use 'table*' and squeeze columns.
        metrics: List of metric keys to include in the table.

    Returns:
        str: A string containing the complete, ready-to-use LaTeX table.
    """
    
    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
        print("Please ensure this utility is available in your environment.")
        # Define a basic placeholder if it's not found, to avoid crashing
        # This assumes a very specific structure and is NOT a real replacement.
        def metrics_aggregation(results, take_error_gold):
            print("Using placeholder 'metrics_aggregation' due to import error.")
            # This is a fallback and will likely fail if data is complex.
            # A more robust solution would be needed if this is a common issue.
            if results:
                return results[0].get('summary', {})
            return {}

    
    # --- 1. Data Collection ---
    # We store data in a flat map for easy lookup
    # Key: (Metric_Type, Exp_Name, Model, SC_Status, Difficulty)
    data_map = {} 
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    for exp_name in sorted_exp_names:
        for model in sorted_models:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue # Skip if model or experiment data is missing
            
            for self_corr in [False, True]:
                self_corr_key = 'self_corrected' if self_corr else 'corrected'
                sc_label = 'W/ Self-Corr' if self_corr else 'W/o Self-Corr'
                
                try:
                    # Get results for this *specific* experiment
                    results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
                except KeyError:
                    # This experiment/model/sc_key combo might not exist
                    continue
                
                if not results_list:
                    continue
                    
                # Aggregate metrics for *this experiment's results only*
                agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
                
                if std_dev == 'run':
                    results_by_diff = agg_metrics.get('by_difficulty_runs', {})
                elif std_dev == 'req_id':
                    results_by_diff = agg_metrics.get('by_difficulty_req_id', {})
                else:
                    results_by_diff = agg_metrics.get('by_difficulty', {}) # Default or 'all'

                for diff, metrics in results_by_diff.items():
                    diff_clean = diff.replace('advanced', 'hard').capitalize()
                    difficulties_found.add(diff_clean)
                    
                    # --- ROW (OID) METRICS ---
                    oid_mean = metrics.get('oids', {}).get(metric_type, 0)
                    oid_std = metrics.get('oids', {}).get(f"{metric_type}_std", 0) if std_dev else 0
                    
                    # --- COLUMN METRICS ---
                    col_metrics = metrics.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
                    col_mean = col_metrics.get(metric_type, 0)
                    col_std = col_metrics.get(f"{metric_type}_std", 0) if std_dev else 0

                    # Format Strings
                    val_fmt = "{:.3f}"
                    if std_dev:
                        val_fmt += " $\pm$ {:.3f}"
                        row_str = val_fmt.format(oid_mean, oid_std)
                        col_str = val_fmt.format(col_mean, col_std)
                    else:
                        row_str = val_fmt.format(oid_mean)
                        col_str = val_fmt.format(col_mean)
                    
                    data_map[(f'ID {metric_type}', exp_name, model, sc_label, diff_clean)] = row_str
                    data_map[(f'Column {metric_type}', exp_name, model, sc_label, diff_clean)] = col_str

    # --- 2. Find Max Values for Bolding (within each metric + experiment group) ---
    
    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)
    sc_labels = ['W/o Self-Corr', 'W/ Self-Corr']
    metric_labels = [f'ID {metric_type}', f'Column {metric_type}']
    
    max_vals = {} # Key: (Metric, Exp_Name, Difficulty, SC_Status) -> max_value
    for metric in metric_labels:
        for exp_name in sorted_exp_names:
            for diff in sorted_diffs:
                for sc in sc_labels:
                    vals = [_get_value(data_map.get((metric, exp_name, model, sc, diff))) for model in sorted_models]
                    max_vals[(metric, exp_name, diff, sc)] = max(vals) if (vals and any(v > -1 for v in vals)) else -1

    # --- 3. Manually Build LaTeX Table String ---
    
    latex_lines = []
    
    # Calculate column counts
    num_data_cols = len(sorted_diffs) * 2
    total_cols = 1 + num_data_cols # 1 for Model stub
    
    # Preamble
    table_env = "table*" if adjust_margins else "table"
    latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
    
    size_commands = {
        'tiny': '\\tiny', 'scriptsize': '\\scriptsize', 'footnotesize': '\\footnotesize',
        'small': '\\small', 'normalsize': '\\normalsize', 'large': '\\large'
    }
    latex_lines.append(size_commands.get(table_size, '\\small'))
    
    latex_lines.append(r"\centering")
    
    if adjust_margins:
        latex_lines.append(r"\setlength{\tabcolsep}{4pt}") 
    
    col_def = "l" + ("c" * num_data_cols)
    latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex_lines.append(r"\toprule")

    # -- Header Row 1: Difficulties (Spanned) --
    header_1 = [""] # Empty for stub
    for diff in sorted_diffs:
        header_1.append(f"\\multicolumn{{2}}{{c}}{{{diff}}}")
    latex_lines.append(" & ".join(header_1) + r" \\")

    # -- Cmidrules for Difficulty Spans --
    cmid_rules = []
    for i in range(len(sorted_diffs)):
        start_col = 2 + (i * 2)
        end_col = 3 + (i * 2)
        cmid_rules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
    latex_lines.append(" ".join(cmid_rules))
    
    # -- Header Row 2: Self-Correction (Decked) --
    header_2 = ["Model"] # Stub header
    for _ in sorted_diffs:
        header_2.append("W/o Self-Corr")
        header_2.append("W/ Self-Corr")
    latex_lines.append(" & ".join(header_2) + r" \\")
    latex_lines.append(r"\midrule")

    # -- Body: Iterate through metrics, experiments, and models --
    for i, metric in enumerate(metric_labels):
        if i > 0:
            latex_lines.append(r"\midrule") # Separator between OID and Column
            
        for j, exp_name in enumerate(sorted_exp_names):
            if j > 0:
                # Add a lighter rule between experiments *within* the same metric block
                latex_lines.append(f"\\cmidrule(lr){{1-{total_cols}}}")
            
            # -- Row Spanner for Metric Type + Experiment --
            metric_label = f"ID {metric_type} (Rows)" if metric.startswith("ID") else f"Column {metric_type}"
            metric_label = metric_label.replace("_", "\_")
            # Clean up exp_name for display (e.g., remove underscores)
            # exp_display = exp_name.replace("_", " ").title()
            exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
            exp_display = 'Direct' if 'direct' in exp_name else exp_display
            spanner_label = f"\\textbf{{{metric_label} -- {exp_display}}}"
            latex_lines.append(f"\\multicolumn{{{total_cols}}}{{l}}{{{spanner_label}}} \\\\")
            
            # -- Data Rows for this Metric + Experiment --
            for model in sorted_models:
                if 'claude-3-7' in model: model_label = 'Claude-3.7'
                elif 'gpt' in model: 
                    model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
                else: model_label = model
                line = [model_label] # Start with model name
                has_data = False
                for diff in sorted_diffs:
                    for sc in sc_labels:
                        val_str = data_map.get((metric, exp_name, model, sc, diff), "-")
                        if val_str != "-":
                            has_data = True
                        val_num = _get_value(val_str)
                        max_val = max_vals.get((metric, exp_name, diff, sc), -1)
                        
                        # Apply bold if it's the max for this group
                        if abs(val_num - max_val) < 1e-6 and max_val >= 0:
                            line.append(f"\\textbf{{{val_str}}}")
                        else:
                            line.append(val_str)
                
                # Only add the row if it contains data
                if has_data:
                    latex_lines.append(" & ".join(line) + r" \\")

    # -- Footer --
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    # Caption and Label
    latex_lines.append(r"\caption{'{metric_type}' by Experiment, Difficulty, and Self-Correction. Best result in each group and column is bolded.}")
    latex_lines.append(r"\label{tab:combined_vertical_results_by_exp}")
    
    latex_lines.append(f"\\end{{{table_env}}}")
    
    final_latex_string = "\n".join(latex_lines)
    
    # Print for user
    print("\n" + "="*80)
    print("Vertically Split LaTeX Table by Experiment (Booktabs Format)")
    print("="*80)
    print(final_latex_string)
    print("="*80)
    
    return final_latex_string



def table_metrics_by_difficulty_models(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    std_dev: Union[str, None] = None,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates separate tables showing precision, recall, and F1-score metrics for rows and columns by difficulty and self-correction status.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                            for models and experiments, and values containing metrics.
        model_names (List[str]): List of model names to include in the table.
        exp_names (List[str]): A list of experiment names to include in the table.
        std_dev (str): The standard deviation type to use for the table. Options are 'run', 'req_id', or 'all'.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.
        take_error_golds (bool): Whether to include gold queries with errors.
        table_size (str): Size of the table. Options: 'tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large'.
        adjust_margins (bool): If True, adds margin adjustment commands to fit table in page.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        Six tables - precision (rows, columns), recall (rows, columns), F1-score (rows, columns).
    """
    
    from llm.utils.eval_utils import metrics_aggregation
    
    # Initialize data collection for each metric
    precision_rows_data = {}
    precision_columns_data = {}
    recall_rows_data = {}
    recall_columns_data = {}
    f1_rows_data = {}
    f1_columns_data = {}
    difficulties = set()
    
    # Collect data for both corrected and self_corrected
    for self_corr in [False, True]:
        self_corr_key = 'self_corrected' if self_corr else 'corrected'
        self_corr_label = 'W/o Self-Correction' if not self_corr else 'W/ Self-Correction'
        
        for model in sorted(model_names):
            if model not in precision_rows_data:
                precision_rows_data[model] = {}
                precision_columns_data[model] = {}
                recall_rows_data[model] = {}
                recall_columns_data[model] = {}
                f1_rows_data[model] = {}
                f1_columns_data[model] = {}
            
            experiments = sorted([exp for exp in evaluation_results[model].keys() if exp in exp_names])
            
            # Aggregate metrics across all experiments for this model
            all_results = []
            for experiment in experiments:
                try:
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    all_results.extend(results)
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment} - {self_corr_key}: {e}")
                    continue
            
            if not all_results:
                continue
            
            # Aggregate metrics across all experiments
            aggregate_metrics = metrics_aggregation(results=all_results, take_error_gold=take_error_golds)
            
            print(f"================================ {model} - {self_corr_label} ================================")
            print(f"Number of gold queries with errors: {aggregate_metrics['errors']['gold_errors']}")
            
            # Use the appropriate metrics based on std_dev setting
            if std_dev == 'all' or std_dev is None:
                results_by_diff = aggregate_metrics['by_difficulty']
            elif std_dev == 'run':
                results_by_diff = aggregate_metrics['by_difficulty_runs']
            elif std_dev == 'req_id':
                results_by_diff = aggregate_metrics['by_difficulty_req_id']
            else:
                raise ValueError(f"Unknown standard deviation type: {std_dev}")
            
            # Extract metrics for each difficulty
            for difficulty, metrics in results_by_diff.items():
                difficulties.add(difficulty)
                
                # OID metrics
                oid_precision = metrics['oids']['precision']
                oid_precision_std = metrics['oids'].get('precision_std', 0) if std_dev else 0
                oid_recall = metrics['oids']['recall']
                oid_recall_std = metrics['oids'].get('recall_std', 0) if std_dev else 0
                oid_f1 = metrics['oids']['f1_score']
                oid_f1_std = metrics['oids'].get('f1_score_std', 0) if std_dev else 0
                
                # Column metrics
                if formatted_columns and 'columns_formatted' in metrics:
                    col_precision = metrics['columns_formatted']['precision']
                    col_precision_std = metrics['columns_formatted'].get('precision_std', 0) if std_dev else 0
                    col_recall = metrics['columns_formatted']['recall']
                    col_recall_std = metrics['columns_formatted'].get('recall_std', 0) if std_dev else 0
                    col_f1 = metrics['columns_formatted']['f1_score']
                    col_f1_std = metrics['columns_formatted'].get('f1_score_std', 0) if std_dev else 0
                else:
                    col_precision = metrics['columns']['precision']
                    col_precision_std = metrics['columns'].get('precision_std', 0) if std_dev else 0
                    col_recall = metrics['columns']['recall']
                    col_recall_std = metrics['columns'].get('recall_std', 0) if std_dev else 0
                    col_f1 = metrics['columns']['f1_score']
                    col_f1_std = metrics['columns'].get('f1_score_std', 0) if std_dev else 0
                
                # Store in separate table data
                difficulty_clean = difficulty.replace('advanced', 'hard')
                col_key = (self_corr_label, difficulty_clean.capitalize())
                
                if std_dev:
                    precision_rows_data[model][col_key] = f"{oid_precision:.3f} ± {oid_precision_std:.3f}"
                    precision_columns_data[model][col_key] = f"{col_precision:.3f} ± {col_precision_std:.3f}"
                    recall_rows_data[model][col_key] = f"{oid_recall:.3f} ± {oid_recall_std:.3f}"
                    recall_columns_data[model][col_key] = f"{col_recall:.3f} ± {col_recall_std:.3f}"
                    f1_rows_data[model][col_key] = f"{oid_f1:.3f} ± {oid_f1_std:.3f}"
                    f1_columns_data[model][col_key] = f"{col_f1:.3f} ± {col_f1_std:.3f}"
                else:
                    precision_rows_data[model][col_key] = f"{oid_precision:.3f}"
                    precision_columns_data[model][col_key] = f"{col_precision:.3f}"
                    recall_rows_data[model][col_key] = f"{oid_recall:.3f}"
                    recall_columns_data[model][col_key] = f"{col_recall:.3f}"
                    f1_rows_data[model][col_key] = f"{oid_f1:.3f}"
                    f1_columns_data[model][col_key] = f"{col_f1:.3f}"
    
    # Sort difficulties
    difficulties = sorted(list(difficulties), key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    
    # Create MultiIndex columns
    columns_index = []
    for self_corr_label in ['W/o Self-Correction', 'W/ Self-Correction']:
        for difficulty in difficulties:
            difficulty_clean = difficulty.replace('advanced', 'hard')
            columns_index.append((self_corr_label, difficulty_clean.capitalize()))
    
    # Create DataFrames with MultiIndex columns
    precision_rows_df = pd.DataFrame.from_dict(precision_rows_data, orient='index')
    precision_columns_df = pd.DataFrame.from_dict(precision_columns_data, orient='index')
    recall_rows_df = pd.DataFrame.from_dict(recall_rows_data, orient='index')
    recall_columns_df = pd.DataFrame.from_dict(recall_columns_data, orient='index')
    f1_rows_df = pd.DataFrame.from_dict(f1_rows_data, orient='index')
    f1_columns_df = pd.DataFrame.from_dict(f1_columns_data, orient='index')
    
    # Reorder columns to match the desired structure
    precision_rows_df = precision_rows_df.reindex(columns=columns_index)
    precision_columns_df = precision_columns_df.reindex(columns=columns_index)
    recall_rows_df = recall_rows_df.reindex(columns=columns_index)
    recall_columns_df = recall_columns_df.reindex(columns=columns_index)
    f1_rows_df = f1_rows_df.reindex(columns=columns_index)
    f1_columns_df = f1_columns_df.reindex(columns=columns_index)
    
    # Create MultiIndex for columns
    multi_columns = pd.MultiIndex.from_tuples(columns_index, names=['Self-Correction', 'Difficulty'])
    precision_rows_df.columns = multi_columns
    precision_columns_df.columns = multi_columns
    recall_rows_df.columns = multi_columns
    recall_columns_df.columns = multi_columns
    f1_rows_df.columns = multi_columns
    f1_columns_df.columns = multi_columns
    
    # Fill NaN values with empty string
    precision_rows_df = precision_rows_df.fillna('')
    precision_columns_df = precision_columns_df.fillna('')
    recall_rows_df = recall_rows_df.fillna('')
    recall_columns_df = recall_columns_df.fillna('')
    f1_rows_df = f1_rows_df.fillna('')
    f1_columns_df = f1_columns_df.fillna('')
    
    # Find best results per column for bold formatting
    def format_best_values(df):
        """Add bold formatting to the best (highest) values in each column"""
        df_formatted = df.copy()
        
        for col in df.columns:
            # Extract numeric values from strings like "0.123 ± 0.045" or "0.123"
            numeric_values = []
            for val in df[col]:
                if val == '':
                    numeric_values.append(-1)  # Use -1 for empty values
                else:
                    # Extract the first number (before ±)
                    try:
                        num = float(val.split(' ±')[0]) if ' ±' in val else float(val)
                        numeric_values.append(num)
                    except ValueError:
                        numeric_values.append(-1)
            
            # Find the maximum value and its indices
            max_val = max(numeric_values)
            if max_val > 0:  # Only format if there are valid values
                max_indices = [i for i, val in enumerate(numeric_values) if val == max_val]
                
                # Add bold formatting to the best values
                for idx in max_indices:
                    original_val = df_formatted.iloc[idx, df_formatted.columns.get_loc(col)]
                    if original_val != '':
                        df_formatted.iloc[idx, df_formatted.columns.get_loc(col)] = f"\\textbf{{{original_val}}}"
        
        return df_formatted
    
    def create_enhanced_latex_table(df_formatted, caption, label, table_size='small', adjust_margins=False):
        """Create enhanced LaTeX table with proper formatting and optional margin adjustment"""
        # Determine number of difficulties (assumes equal columns for each self-correction method)
        n_difficulties = len(df_formatted.columns) // 2
        
        # Create column format with vertical line between W/o and W/ self-correction (no line at the end)
        col_format = '|c|' + 'c' * n_difficulties + '|' + 'c' * n_difficulties + '|'
        
        # Generate LaTeX table
        latex_str = df_formatted.to_latex(
            escape=False, 
            multicolumn_format='c',
            column_format=col_format,
            caption=caption,
            label=label,
            position='htbp'
        )
        
        # Add table size and centering
        size_commands = {
            'tiny': '\\tiny',
            'scriptsize': '\\scriptsize', 
            'footnotesize': '\\footnotesize',
            'small': '\\small',
            'normalsize': '\\normalsize',
            'large': '\\large'
        }
        
        size_cmd = size_commands.get(table_size, '\\small')
        
        # Prepare margin adjustment commands
        margin_start = ""
        margin_end = ""
        if adjust_margins:
            margin_start = """\\adjustbox{width=\\textwidth,center}{
"""
            margin_end = """
}"""
        
        # Replace formatting
        latex_str = latex_str.replace('\\begin{tabular}', f'{size_cmd}\n\\centering\n{margin_start}\\begin{{tabular}}')
        latex_str = latex_str.replace('\\end{tabular}', f'\\end{{tabular}}{margin_end}')
        latex_str = latex_str.replace('\\toprule', '\\hline')
        latex_str = latex_str.replace('\\bottomrule', '\\hline')
        
        # Process lines to add hline between self-correction and difficulty rows
        lines = latex_str.split('\n')
        new_lines = []
        header_count = 0
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            # Count header rows and add hline between first and second header row
            if ('W/o Self-Correction' in line and 'W/ Self-Correction' in line) or \
               any(difficulty in line for difficulty in ['Simple', 'Medium', 'Hard']):
                header_count += 1
                # Add hline after first header row (self-correction row)
                if header_count == 1:
                    new_lines.append('\\hline')
        
        # Remove midrule replacements
        latex_str_modified = '\n'.join(new_lines)
        latex_str_modified = latex_str_modified.replace('\\midrule', '')
        
        # Add adjustbox package note if margins are adjusted
        if adjust_margins:
            package_note = "% Note: Add \\usepackage{adjustbox} to your LaTeX preamble for margin adjustment\n"
            latex_str_modified = package_note + latex_str_modified
        
        return latex_str_modified
    
    # Format best values with bold
    precision_rows_df_formatted = format_best_values(precision_rows_df)
    precision_columns_df_formatted = format_best_values(precision_columns_df)
    recall_rows_df_formatted = format_best_values(recall_rows_df)
    recall_columns_df_formatted = format_best_values(recall_columns_df)
    f1_rows_df_formatted = format_best_values(f1_rows_df)
    f1_columns_df_formatted = format_best_values(f1_columns_df)
    
    if "sbs" in exp_names[0]: exp_label = "using step-by-step generation method"
    if "dir" in exp_names[0]: exp_label = "using direct generation method"
    # Print tables in LaTeX format with enhanced formatting
    print("\n" + "="*80)
    print("PRECISION - ROWS - LaTeX Format")
    print("="*80)
    latex_str_precision_rows = create_enhanced_latex_table(
        precision_rows_df_formatted, 
        f'Precision Scores for Rows {exp_label}',
        'tab:precision_rows',
        table_size,
        adjust_margins
    )
    print(latex_str_precision_rows)

    print("\n" + "="*80)
    print("PRECISION - COLUMNS - LaTeX Format") 
    print("="*80)
    latex_str_precision_columns = create_enhanced_latex_table(
        precision_columns_df_formatted,
        f'Precision Scores for Columns {exp_label}', 
        'tab:precision_columns',
        table_size,
        adjust_margins
    )
    print(latex_str_precision_columns)

    print("\n" + "="*80)
    print("RECALL - ROWS - LaTeX Format")
    print("="*80)
    latex_str_recall_rows = create_enhanced_latex_table(
        recall_rows_df_formatted, 
        f'Recall Scores for Rows {exp_label}',
        'tab:recall_rows',
        table_size,
        adjust_margins
    )
    print(latex_str_recall_rows)

    print("\n" + "="*80)
    print("RECALL - COLUMNS - LaTeX Format") 
    print("="*80)
    latex_str_recall_columns = create_enhanced_latex_table(
        recall_columns_df_formatted,
        f'Recall Scores for Columns {exp_label}', 
        'tab:recall_columns',
        table_size,
        adjust_margins
    )
    print(latex_str_recall_columns)

    print("\n" + "="*80)
    print("F1-SCORE - ROWS - LaTeX Format")
    print("="*80)
    latex_str_f1_rows = create_enhanced_latex_table(
        f1_rows_df_formatted, 
        f'F1-Scores for Rows {exp_label}',
        'tab:f1_rows',
        table_size,
        adjust_margins
    )
    print(latex_str_f1_rows)

    print("\n" + "="*80)
    print("F1-SCORE - COLUMNS - LaTeX Format") 
    print("="*80)
    latex_str_f1_columns = create_enhanced_latex_table(
        f1_columns_df_formatted,
        f'F1-Scores for Columns {exp_label}', 
        'tab:f1_columns',
        table_size,
        adjust_margins
    )
    print(latex_str_f1_columns)

    print("\n" + "="*80)
    print("Available table sizes: tiny, scriptsize, footnotesize, small, normalsize, large")
    print("Current table size:", table_size)
    print("Margin adjustment:", "enabled" if adjust_margins else "disabled")
    print("="*80)
    
    return (precision_rows_df_formatted, precision_columns_df_formatted, 
            recall_rows_df_formatted, recall_columns_df_formatted, 
            f1_rows_df_formatted, f1_columns_df_formatted)



# This function is a modified version of the original plot_perfect_match_by_difficulty_model_sc
# to compare the self-corrected and corrected results for a specific model and experiments.
def plot_perfect_match_by_difficulty_model_sc(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: List[str],
        std_dev: Union[str, None] = None,
        formatted_columns: bool = True,
        take_error_golds: bool = False,
        save_fig: Union[str, None] = None
) -> None:
    """
    Plots the perfect match rates for OID and column matches by difficulty from evaluation results for a specific model and experiments,
    comparing self-corrected and corrected results.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_names (List[str]): A list of experiment names to include in the plot.
        std_dev (str): The standard deviation type to use for the plot. Options are 'run', 'req_id', or 'all'.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays a plot but does not return any value.
        
    Note:
        The expected structure of evaluation_results is:
        {
            'model_name': {
                'experiment_name': {
                    'self_corrected': {
                        'aggregate_metrics': {
                            'oids': {'perfect_match_rate': float},
                            'columns': {'perfect_match_rate': float},
                            'difficulty': str  # e.g., 'easy', 'medium', 'hard'
                        }
                    }
                }
            }
        }
    """
    
    # Initialize data collection lists
    experiment_labels = {}
    difficulties = []
    
    # Self-corrected results are expected
    # if self_corr: self_corr_key = 'self_corrected'
    # else: self_corr_key = 'corrected'

    from llm.utils.eval_utils import metrics_aggregation

    # Collect data for plotting
    experiments = [exp for exp in evaluation_results[model_name].keys() if exp in exp_names]
    for self_corr_key in [ 'corrected', 'self_corrected',]:
        experiment_labels[self_corr_key] = {}
        for experiment in experiments:
        # Check if the experiment is self-corrected or corrected
        # Iterate through self-corrected and corrected results
            try:
                if "dir" in experiment:
                    exp_label = "Direct"
                if "sbs" in experiment:
                    exp_label = "Step-by-Step"
                
                oid_match_rates = {}
                column_match_rates = {}
                if std_dev:
                    oid_match_rates_std = {}
                    column_match_rates_std = {}

                results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
                aggregate_metrics = metrics_aggregation(results=results, take_error_gold=take_error_golds)
                # ['errors']['gold_errors']
                print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                # Use the aggregate metrics directly
                if std_dev == 'all' or std_dev is None:
                    # results = evaluation_results[model][experiment][self_corr_key]['aggregate_metrics']['by_difficulty']
                    results = aggregate_metrics['by_difficulty']
                elif std_dev == 'run':
                    results = aggregate_metrics['by_difficulty_runs']
                elif std_dev == 'req_id':
                    results = aggregate_metrics['by_difficulty_req_id']
                else:
                    raise ValueError(f"Unknown standard deviation type: {std_dev}")
                # iterate through difficulties
                for difficulty, metrics in results.items():
                    if difficulty not in difficulties:
                        difficulties.append(difficulty)
                    oid_match_rates[difficulty] = metrics['oids']['perfect_match_rate']
                    if std_dev: oid_match_rates_std[difficulty] = metrics['oids']['perfect_match_rate_std']
                    if formatted_columns and 'columns_formatted' in metrics:
                        column_match_rates[difficulty] = metrics['columns_formatted']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns_formatted']['perfect_match_rate_std']
                    else:
                        column_match_rates[difficulty] = metrics['columns']['perfect_match_rate']
                        if std_dev: column_match_rates_std[difficulty] = metrics['columns']['perfect_match_rate_std']
                # Add standard deviation to the match rates
                if std_dev:
                    experiment_labels[self_corr_key][exp_label] = {
                        'oid': oid_match_rates,
                        'oid_std': oid_match_rates_std,
                        'column': column_match_rates,
                        'column_std': column_match_rates_std
                    }
                else:
                    experiment_labels[self_corr_key][exp_label] = {
                        'oid': oid_match_rates,
                        'column': column_match_rates
                    }
            except KeyError as e:
                print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
                
    # Figure with two subplots: OID and Column matches
    # Each subplot will be separated by difficulty on the x-axis
    # Will be bar plots for each difficulty by experiment
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    # difficulties = list(next(iter(experiment_labels.values()))['oid'].keys())
    # order difficulties: simple, medium, advanced
    difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    x = np.arange(len(difficulties))  # the label locations
    width = 0.3  # the width of the bars
    stds = []
    for i, (self_corr_key, data) in enumerate(experiment_labels.items()):
        stds_ = []
        stds__ = []
        for j, (exp_label, exp_data) in enumerate(sorted(data.items())):
            oid_rates = [exp_data['oid'][d] for d in difficulties]
            column_rates = [exp_data['column'][d] for d in difficulties]
            # Plot OID matches
            axs[0, i].bar(x + (j - 0.5) * width, oid_rates, width, label=exp_label)
            # If std_dev is used, add error bars
            # Plot Column matches
            axs[1, i].bar(x + (j - 0.5) * width, column_rates, width, label=exp_label, )
            if std_dev:
                oid_std = [exp_data['oid_std'][d] for d in difficulties]
                stds_.extend(oid_std)
                axs[0, i].errorbar(x + (j - 0.5) * width, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5)
                column_std = [exp_data['column_std'][d] for d in difficulties]
                stds__.extend(column_std)
                axs[1, i].errorbar(x + (j - 0.5) * width, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5)
                # for n in range(len(column_std)):
                #     stds_.append(oid_std[n])
                #     stds_.append(column_std[n])
            # stds__.append(stds_)
        # Store the standard deviations for later use
        # stds.extend(stds__)  # Mean of the standard deviations for each subplot
        # stds.append(stds__)  # Mean of the standard deviations for each subplot
        stds.append(stds_)
        stds.append(stds__)

    # Set x-ticks and labels
    axs[0, 0].set_ylabel('Rows', fontsize=20)
    axs[1, 0].set_ylabel('Columns', fontsize=20)
    
    axs[1, 0].set_xticks(x)
    axs[1, 1].set_xticks(x)
    # change advanced to hard
    difficulties = [d.replace('advanced', 'hard').capitalize() for d in difficulties]
    axs[1, 0].set_xticklabels(difficulties, fontsize=18)
    axs[1, 1].set_xticklabels(difficulties, fontsize=18)
    axs[0, 0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1, 0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0, 1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1, 1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    # Remove numbers from the y-axis of the second column
    axs[0, 1].yaxis.set_visible(False)
    axs[1, 1].yaxis.set_visible(False)
    # remove the 0 value from the y-axis of the first plot
    axs[0, 0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])    
    # Set x label by each column
    # axs[0, 0].set_title('Without Self-Correction', fontsize=18)
    # axs[0, 1].set_title('With Self-Correction', fontsize=18)
    axs[1, 0].set_xlabel('Without Self-Correction', fontsize=24)
    axs[1, 1].set_xlabel('With Self-Correction', fontsize=24)

    # add legend to the first column
    axs[0, 1].legend(bbox_to_anchor=(1.01, 1.0))
    # Remove space between the two rows of subplots
    plt.subplots_adjust(hspace=0.0, wspace=0.01)

    # set suptitle
    plt.suptitle('Query Generation Strategies: Self-Correction vs. No Self-Correction', fontsize=24, y=0.95)
    # Add values on top of the bars
    for j, ax in enumerate(axs.flatten()):
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # Add the value on top of the bar
            # add the value on top of the bar, just above the standard deviation bar height
            ax.text(bar.get_x() + bar.get_width()/2, height +0.01+ stds[j][i], f"{height:.2f}", ha='center', va='bottom')
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.05)
    # plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.01, 1.0),)
    for ax in axs.flatten():
        ax.grid(True, which='both')
    plt.grid(True)

    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()

# plot 3 plots, one for each difficulty level, showing the perfect match rates for OID and column matches by req_id
def plot_perfect_match_by_difficulty_req_id_model(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_name: str,
        self_corr: bool = True,
        formatted_columns: bool = True
) -> None:
    # Extract relevant data
    req_ids = []
    oid_matches = []
    column_matches = []

    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'
    from llm.utils.eval_utils import metrics_aggregation
    
    try:
        # exp_data = evaluation_results[model_name][exp_name]
        results = evaluation_results[model_name][exp_name][self_corr_key]['detailed_results']
        aggregate_metrics = metrics_aggregation(results=results)
        results = aggregate_metrics['by_req_id']

        for req_id, metrics in results.items():
            req_ids.append(req_id)
            oid_matches.append(metrics['oids']['perfect_match_rate'])
            if formatted_columns and 'columns_formatted' in metrics:
                column_matches.append(metrics['columns_formatted']['perfect_match_rate'])
            else:
                column_matches.append(metrics['columns']['perfect_match_rate'])

            
    except KeyError as e:
        print(f"Warning: Missing data for {model_name} - {exp_name}: {e}")

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'req_id': req_ids,
        'oid_match': oid_matches,
        'column_match': column_matches
    })

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(10, 15), sharex=True)
    sns.boxplot(data=df, x='req_id', y='oid_match', ax=axs[0, 0])
    sns.boxplot(data=df, x='req_id', y='column_match', ax=axs[0, 1])
    sns.boxplot(data=df, x='req_id', y='column_match', ax=axs[1, 0])
    sns.boxplot(data=df, x='req_id', y='column_match', ax=axs[1, 1])
    sns.boxplot(data=df, x='req_id', y='column_match', ax=axs[2, 0])
    sns.boxplot(data=df, x='req_id', y='column_match', ax=axs[2, 1])

    axs[0, 0].set_title('OID Match by Request ID')
    axs[0, 1].set_title('Column Match by Request ID')
    axs[1, 0].set_title('Column Match by Request ID')
    axs[1, 1].set_title('Column Match by Request ID')
    axs[2, 0].set_title('Column Match by Request ID')
    axs[2, 1].set_title('Column Match by Request ID')

    plt.tight_layout()
    plt.show()


def get_error_class(error_message: str) -> dict:
    """
    Extract the error class and error message from an ALeRCE's database execution error.
    This function is designed to handle errors from the psycopg2 library, which is commonly used for PostgreSQL database interactions.
    
    Args:
        error_message (str): The error string from the database.
        
    Returns:
        dict: A dictionary containing the error type, error message, and error class.
    """
    import sqlparse
    import re
    from llm.constants import SQLErrorType
    error_class = None
    error_line = None
    
    # Extract the error message
    if "psycopg2.errors" in error_message:
        # Extract the error class between "psycopg2.errors." and ")"
        error_match = re.search(r"psycopg2\.errors\.([^)]*)", error_message)
        if error_match:
            error_class = error_match.group(1)
        
        # Extract the line where the error occurred
        error_line_match = re.search(r"\)\s*([^\\]*?)(?:\n\[SQL:|$)", error_message)
        if error_line_match:
            error_line = error_line_match.group(1).strip()
    
    # Determine error type based on content
    if 'timeout' in error_message.lower():
        error_type = SQLErrorType.TIMEOUT
    elif 'not exist' in error_message.lower() or 'does not exist' in error_message.lower():
        error_type = SQLErrorType.UNDEFINED
    else:
        error_type = SQLErrorType.OTHER
    
    return {
        'error_type': error_type,
        'error_message': error_message,
        'error_class': error_class,
        'error_line': error_line
    }


def plot_execution_errors_by_model(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: str,
        percentage: bool = False,
        save_fig: Union[str, None] = None,
) -> None:
    """
    Plots the execution errors by model and experiment.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_name (str): The name of the experiment to plot results for.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.

    Returns:
        None: This function displays a plot but does not return any value.
    """
    
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    experiment_labels = {}
    for self_corr_key in [ 'corrected', 'self_corrected',]:
        experiment_labels[self_corr_key] = {}
        for experiment in experiments:
        # Check if the experiment is self-corrected or corrected
        # Iterate through self-corrected and corrected results
            try:
                if "dir" in experiment:
                    exp_label = "Direct"
                if "sbs" in experiment:
                    exp_label = "Step-by-Step"
                results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
                error_types = []
                for res in results:
                    error_info = res['comparison'].get('error_pred', None)
                    if error_info:
                        error_types.append(get_error_class(error_info).get('error_class'))
                    else:
                        error_types.append(0)  # No error
                        continue
                
                # Store the error types for each experiment
                experiment_labels[self_corr_key][exp_label] = error_types
            
            except KeyError as e:
                print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    print(experiment_labels)

    fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
    for i, (self_corr_key, data) in enumerate(experiment_labels.items()):
        # group the error types by their class and by experiment, to plot them without stacking
        for j, (exp_label, error_list) in enumerate(data.items()):
            error_counts = pd.Series(error_list).value_counts().reset_index()
            # Change "QueryCanceled" to "Timeout"
            error_counts['index'] = error_counts['index'].replace({
                'QueryCanceled': 'Timeout',})
            error_counts.columns = ['error_type', 'error_count']
            error_counts['experiment'] = exp_label
            if j == 0:
                error_data = error_counts
            else:
                error_data = pd.concat([error_data, error_counts], ignore_index=True)
            
        if percentage:
            total_errors = error_data['error_count'].sum()
            error_data['error_count'] = (error_data['error_count'] / total_errors) * 100
        # remove rows with error_type 0
        error_data = error_data[error_data['error_type'] != 0]
                
        sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
        if percentage:
            # add value labels on top of each bar
            for p in axs[i].patches:
                height = p.get_height()
                if height > 0:
                    axs[i].annotate(f'{height:.1f}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center',
                                    xytext=(0, 9),
                                    textcoords='offset points',
                                    fontsize=12)
        else:
            for p in axs[i].patches:
                height = p.get_height()
                if height > 0:
                    axs[i].annotate(f'{int(height)}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center',
                                    xytext=(0, 9),
                                    textcoords='offset points',
                                    fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
        # remove legend title
        axs[i].legend_.set_title(None)
    axs[0].set_xlabel("Without Self-Correction", fontsize=22)
    if percentage: 
        axs[0].set_ylabel("Percentage from Total Queries (%)", fontsize=24)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
    else: axs[0].set_ylabel("Error Count", fontsize=24)
    axs[1].set_xlabel("With Self-Correction", fontsize=22)
    axs[0].yaxis.set_visible(True)
    axs[1].yaxis.set_visible(True)
    plt.legend(bbox_to_anchor=(1.0, 1.0), fontsize=12)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=65, ha='right', fontsize=18)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=65, ha='right', fontsize=18)
    axs[0].grid(True)
    axs[1].grid(True)
    
    plt.suptitle(f"Execution Errors by Generation Method", fontsize=24, y=0.965)
    plt.ylabel("Error Count", fontsize=24)
    
    plt.subplots_adjust(hspace=0.1, wspace=0.02)
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.tight_layout()
    axs[0].grid(True)
    axs[1].grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    plt.show()

def plot_execution_errors_by_model_by_difficulty(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: str,
        self_corr: bool = True,
        percentage: bool = False,
        save_fig: Union[str, None] = None,
) -> None:
    """
    Plots the execution errors by model and experiment.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_names (str): The name of the experiments to plot results for.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
        percentage (bool): If True, shows percentages instead of counts.

    Returns:
        None: This function displays a plot but does not return any value.
    """
    
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    if self_corr: 
        self_corr_key = 'self_corrected'
    else: 
        self_corr_key = 'corrected'
    
    experiment_labels = {}
    total_queries_per_exp_diff = {}  # Track total queries per experiment and difficulty for percentage calculation
    
    for experiment in experiments:
        try:
            if "dir" in experiment:
                exp_label = "Direct"
            elif "sbs" in experiment:
                exp_label = "Step-by-Step"
            else:
                exp_label = experiment
            
            results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
            
            experiment_labels[exp_label] = {}
            total_queries_per_exp_diff[exp_label] = {}
            
            for res in results:
                error_info = res['comparison'].get('error_pred', None)
                difficulty = res.get('difficulty')
                
                if difficulty not in experiment_labels[exp_label]:
                    experiment_labels[exp_label][difficulty] = []
                    total_queries_per_exp_diff[exp_label][difficulty] = 0
                
                total_queries_per_exp_diff[exp_label][difficulty] += 1
                
                if error_info:
                    error_class = get_error_class(error_info).get('error_class')
                    experiment_labels[exp_label][difficulty].append(error_class)
        
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    
    print(experiment_labels)
    
    difficulties = sorted(set(
        difficulty for data in experiment_labels.values() for difficulty in data.keys()
    ), key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    
    # Handle case where there's only one difficulty (can't use indexing on single subplot)
    if len(difficulties) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axs = [ax]
    else:
        fig, axs = plt.subplots(1, len(difficulties), figsize=(6 * len(difficulties), 8), sharey=True, sharex=True)
        if len(difficulties) == 2:
            axs = list(axs)  # Ensure axs is always a list
    
    for i, difficulty in enumerate(difficulties):
        error_data_combined = []
        
        for exp_label, data in experiment_labels.items():
            if difficulty in data and data[difficulty]:  # Only process if there are errors
                error_counts = pd.Series(data[difficulty]).value_counts().reset_index()
                # Change "QueryCanceled" to "Timeout"
                error_counts['index'] = error_counts['index'].replace({
                    'QueryCanceled': 'Timeout'})
                error_counts.columns = ['error_type', 'error_count']
                error_counts['experiment'] = exp_label
                
                if percentage:
                    total_queries = total_queries_per_exp_diff[exp_label][difficulty]
                    error_counts['error_count'] = (error_counts['error_count'] / total_queries) * 100
                
                error_data_combined.append(error_counts)
        
        # Plot the errors for each difficulty
        if not error_data_combined:
            print(f"No errors found for difficulty {difficulty} in model {model_name}.")
            axs[i].text(0.5, 0.5, 'No Errors', horizontalalignment='center', 
                       verticalalignment='center', transform=axs[i].transAxes, fontsize=16)
            axs[i].set_xlabel(f"{difficulty.replace('advanced','hard').title()}", fontsize=16)
            continue
        
        error_data = pd.concat(error_data_combined, ignore_index=True)
        
        # Create the bar plot
        sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
        
        # Add value labels on top of each bar
        for p in axs[i].patches:
            height = p.get_height()
            if height > 0:
                if percentage:
                    axs[i].annotate(f'{height:.1f}%',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='bottom',
                                    xytext=(0, 3),
                                    textcoords='offset points',
                                    fontsize=10)
                else:
                    axs[i].annotate(f'{int(height)}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='bottom',
                                    xytext=(0, 3),
                                    textcoords='offset points',
                                    fontsize=10)
        
        # Set labels and formatting - put difficulty label at bottom as xlabel
        axs[i].set_xlabel(f"{difficulty.replace('advanced','hard').title()}", fontsize=20)
        axs[i].set_title("")  # Remove title from top
        
        if i == 0:  # Only show y-label on first subplot
            if percentage: 
                axs[i].set_ylabel("Error Percentage (%)", fontsize=24)
            else: 
                axs[i].set_ylabel("Error Count", fontsize=24)
        else:
            axs[i].set_ylabel("")
        
        # Rotate x-axis labels for better readability
        axs[i].tick_params(axis='x', rotation=65, labelsize=16)
        axs[i].tick_params(axis='y', labelsize=12)
        
        # Add legend only to the last subplot
        if i == len(difficulties) - 1:
            axs[i].legend(bbox_to_anchor=(0.65, 1), loc='upper left', fontsize=12)
        else:
            axs[i].get_legend().remove()
    
    if percentage: 
        axs[0].set_ylabel("Percentage from Total Queries (%)", fontsize=18)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
        axs[2].set_ylim(0, 100)
    # Set main title and layout
    if self_corr:
        corr_text = "With Self-Correction"
    else:
        corr_text = "Without Self-Correction"
    fig.suptitle(f"Execution Errors by Difficulty - {model_name} ({corr_text})", fontsize=24)
    plt.tight_layout()
    for ax in axs:
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    # reduce space between subplots
    plt.subplots_adjust(wspace=0.03)
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()

def plot_execution_errors_by_model_type(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: str,
        percentage: bool = False,
        save_fig: Union[str, None] = None,
) -> None:
    """
    Plots the execution errors by model and experiment.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        model_name (str): The name of the model to plot results for.
        exp_name (str): The name of the experiment to plot results for.
        self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.

    Returns:
        None: This function displays a plot but does not return any value.
    """
    
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    experiment_labels = {}
    for self_corr_key in [ 'corrected', 'self_corrected',]:
        experiment_labels[self_corr_key] = {}
        for experiment in experiments:
        # Check if the experiment is self-corrected or corrected
        # Iterate through self-corrected and corrected results
            try:
                if "dir" in experiment:
                    exp_label = "Direct"
                if "sbs" in experiment:
                    exp_label = "Step-by-Step"
                results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
                error_types = []
                for res in results:
                    error_info = res['comparison'].get('error_pred', None)
                    if error_info:
                        error_types.append(get_error_class(error_info).get('error_type').upper())
                    else:
                        error_types.append(0)  # No error
                        continue
                
                # Store the error types for each experiment
                experiment_labels[self_corr_key][exp_label] = error_types
            
            except KeyError as e:
                print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
    for i, (self_corr_key, data) in enumerate(experiment_labels.items()):
        for j, (exp_label, error_list) in enumerate(data.items()):
            error_counts = pd.Series(error_list).value_counts().reset_index()
            error_counts.columns = ['error_type', 'error_count']
            error_counts['experiment'] = exp_label
            if j == 0:
                error_data = error_counts
            else:
                error_data = pd.concat([error_data, error_counts], ignore_index=True) 
        if percentage:
            total_errors = error_data['error_count'].sum()
            error_data['error_count'] = (error_data['error_count'] / total_errors) * 100
        # remove rows with error_type 0
        error_data = error_data[error_data['error_type'] != 0]
        sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
        if percentage:
            # add value labels on top of each bar
            for p in axs[i].patches:
                if p.get_height() > 0:
                    height = p.get_height()
                    axs[i].annotate(f'{height:.1f}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center',
                                    xytext=(0, 9),
                                    textcoords='offset points',
                                    fontsize=12)
        else:
            for p in axs[i].patches:
                if p.get_height() > 0:
                    height = p.get_height()
                    axs[i].annotate(f'{int(height)}',
                                    (p.get_x() + p.get_width() / 2., height),
                                    ha='center', va='center',
                                    xytext=(0, 9),
                                    textcoords='offset points',
                                    fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        # plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
        # Add legend only to the last subplot
        if i == len(axs) - 1:
            axs[i].legend(bbox_to_anchor=(0.75, 1), loc='upper left', fontsize=12)
        else:
            axs[i].get_legend().remove()
    axs[0].set_xlabel("Without Self-Correction", fontsize=22)
    if percentage: 
        axs[0].set_ylabel("Percentage from Total Queries (%)", fontsize=24)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
    else: axs[0].set_ylabel("Error Count", fontsize=24)
    axs[1].set_xlabel("With Self-Correction", fontsize=22)
    axs[1].yaxis.set_visible(True)
    plt.subplots_adjust(hspace=0.02, )
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    plt.suptitle(f"Execution Errors Type by Generation Method", fontsize=24)
    plt.ylabel("Error Count", fontsize=24)
    plt.tight_layout()
    
    for ax in axs:
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()


def plot_percentage_errors_by_model(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_name: str,
    exp_names: str,
    formatted_columns: bool = True,
    percentage: bool = False,
    save_fig: Union[str, None] = None,
) -> None:
    """
    Plots the percentage between correct queries (n_perfect_match = 1), queries with semantic errors (n_perfect_match != 0 but without execution error),
    and queries with execution errors, by model and experiment.

    Args:
    evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                    for models and experiments, and values containing metrics.
    model_name (str): The name of the model to plot results for.
    exp_name (str): The name of the experiment to plot results for.
    self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.

    Returns:
    None: This function displays a plot but does not return any value.
    """
    
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    experiment_labels = {}
    for self_corr_key in [ 'corrected', 'self_corrected',]:
        experiment_labels[self_corr_key] = {}
        for experiment in experiments:
        # Check if the experiment is self-corrected or corrected
        # Iterate through self-corrected and corrected results
            try:
                if "dir" in experiment:
                    exp_label = "Direct"
                if "sbs" in experiment:
                    exp_label = "Step-by-Step"
                results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
                error_types_oids = []
                error_types_columns = []
                for res in results:
                    error_info = res['comparison'].get('error_pred', None)
                    if error_info:
                        error_types_oids.append(2)  # Execution error
                        error_types_columns.append(2)  # Execution error
                    else:
                        if res['comparison']['oids']['perfect_match'] == 1:
                            error_types_oids.append(0)  # Perfect match
                        else:
                            error_types_oids.append(1)  # Semantic error
                        if formatted_columns and res['comparison']['columns_formatted']['perfect_match'] == 1:
                            error_types_columns.append(0)  # Perfect match
                        elif not formatted_columns and res['comparison']['columns']['perfect_match'] == 1:
                            error_types_columns.append(0)  # Perfect match
                        else:
                            error_types_columns.append(1)  # Semantic error
                
                # Store the error types for each experiment
                experiment_labels[self_corr_key][exp_label] = {
                    'oids': error_types_oids,
                    'columns': error_types_columns
                }
            
            except KeyError as e:
                print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    
    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    
    # Define error type labels for better visualization
    error_labels = {0: 'Perfect Match', 1: 'Semantic Error', 2: 'Execution Error'}
    
    # Plot for each self-correction key and metric type
    for i, (self_corr_key, data) in enumerate(experiment_labels.items()):
        for metric_idx, metric_type in enumerate(['oids', 'columns']):
            # Combine data for all experiments for this metric type
            combined_data = []
            
            for exp_label, error_dict in data.items():
                error_list = error_dict[metric_type]
                error_counts = pd.Series(error_list).value_counts().reset_index()
                error_counts.columns = ['error_type', 'error_count']
                error_counts['experiment'] = exp_label
                
                if percentage:
                    total_count = len(error_list)
                    error_counts['error_count'] = (error_counts['error_count'] / total_count) * 100
                
                combined_data.append(error_counts)
                
                if combined_data:
                    error_data = pd.concat(combined_data, ignore_index=True)
                
                # Map error types to labels
                error_data['error_type'] = error_data['error_type'].map(error_labels)
            
            # Create the bar plot
            sns.barplot(x='error_type', y='error_count', data=error_data, 
                    ax=axs[metric_idx, i], hue='experiment')
            
            # Add value labels on top of each bar
            for p in axs[metric_idx, i].patches:
                height = p.get_height()
                if height > 0:
                    if percentage:
                        axs[metric_idx, i].annotate(f'{height:.1f}%',
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 3),
                                        textcoords='offset points',
                                        fontsize=10)
                    else:
                        axs[metric_idx, i].annotate(f'{int(height)}',
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 3),
                                        textcoords='offset points',
                                        fontsize=10)
            
            # Set titles and labels
            if metric_idx == 1:  # second row (oids)
                if i == 0:
                    axs[metric_idx, i].set_xlabel("Without Self-Correction", fontsize=20)
                else:
                    axs[metric_idx, i].set_xlabel("With Self-Correction", fontsize=20)
            
            if metric_idx == 0:
                axs[metric_idx, i].set_ylabel("Rows", fontsize=20)
            else:
                axs[metric_idx, i].set_ylabel("Columns", fontsize=20)
            
            # Rotate x-axis labels
            # axs[metric_idx, i].tick_params(axis='x', rotation=45, labelsize=10)
            
            # Add legend only to the top-right subplot
            if i == 1:
                axs[metric_idx, i].legend(bbox_to_anchor=(0.6, 1), loc='upper left', fontsize=16)
            else:
                if axs[metric_idx, i].get_legend():
                    axs[metric_idx, i].get_legend().remove()
    
    # Adjust layout
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.035, wspace=0.02)
    # Set y-axis label
    if percentage:
        # plt.ylabel("Percentage (%)", fontsize=14, )
        fig.supylabel('Percentage (%)', fontsize=24, x=0.055)

        # set limit to 100
        for ax in axs.flatten():
            ax.set_ylim(0, 100)
    else:
        plt.ylabel("Count", fontsize=24)
    # set x-axis labels fontsize
    for ax in axs.flatten():
        ax.tick_params(axis='x', labelsize=16)
    # Set main title
    if 'claude-3-7' in model_name: model_label = 'Claude-3.7'
    elif 'gpt' in model_name: 
        model_label = "-".join(model_name.split('-')[0:2]).replace('gpt', 'GPT')
    else: model_label = model_name
    plt.suptitle(f"Query Results Distribution by Generation Method - {model_label}", fontsize=24, y=0.92)
    
    for ax in axs.flatten():
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()



def plot_percentage_errors_by_model_by_difficulty(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_name: str,
    exp_names: str,
    self_corr: bool = True,
    formatted_columns: bool = True,
    percentage: bool = False,
    save_fig: Union[str, None] = None,
) -> None:
    """
    Plots the percentage between correct queries (n_perfect_match = 1), queries with semantic errors (n_perfect_match != 0 but without execution error),
    and queries with execution errors, by model and experiment.

    Args:
    evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                    for models and experiments, and values containing metrics.
    model_name (str): The name of the model to plot results for.
    exp_names (str): The names of the experiments to plot results for.
    self_corr (bool): If True, uses self-corrected results; otherwise uses corrected results.
    formatted_columns (bool): If True, uses formatted columns for comparison; otherwise uses raw columns.
    percentage (bool): If True, shows percentages instead of counts.
    save_fig (Union[str, None]): If provided, saves the figure to the given path.

    Returns:
    None: This function displays a plot but does not return any value.
    """
    
    experiments = sorted([exp for exp in evaluation_results[model_name].keys() if exp in exp_names])
    if self_corr: 
        self_corr_key = 'self_corrected'
    else: 
        self_corr_key = 'corrected'
    
    experiment_labels = {}
    total_queries_per_exp_diff = {}  # Track total queries per experiment and difficulty for percentage calculation

    for experiment in experiments:
    # Check if the experiment is self-corrected or corrected
    # Iterate through self-corrected and corrected results
        try:
            if "dir" in experiment:
                exp_label = "Direct"
            if "sbs" in experiment:
                exp_label = "Step-by-Step"
            results = evaluation_results[model_name][experiment][self_corr_key]['detailed_results']
            
            experiment_labels[exp_label] = {}
            total_queries_per_exp_diff[exp_label] = {}
            for res in results:
                error_info = res['comparison'].get('error_pred', None)
                difficulty = res.get('difficulty')
                
                if difficulty not in experiment_labels[exp_label]:
                    experiment_labels[exp_label][difficulty] = {}
                    experiment_labels[exp_label][difficulty] = {}
                    experiment_labels[exp_label][difficulty]['oids'] = []
                    experiment_labels[exp_label][difficulty]['columns'] = []
                    total_queries_per_exp_diff[exp_label][difficulty] = 0
                
                total_queries_per_exp_diff[exp_label][difficulty] += 1

                if error_info:
                    experiment_labels[exp_label][difficulty]['oids'].append(2)  # Execution error
                    experiment_labels[exp_label][difficulty]['columns'].append(2)  # Execution error
                else:
                    if res['comparison']['oids']['perfect_match'] == 1:
                        experiment_labels[exp_label][difficulty]['oids'].append(0)  # Perfect match
                    else:
                        experiment_labels[exp_label][difficulty]['oids'].append(1)  # Semantic error
                    if formatted_columns and res['comparison']['columns_formatted']['perfect_match'] == 1:
                        experiment_labels[exp_label][difficulty]['columns'].append(0)  # Perfect match
                    elif not formatted_columns and res['comparison']['columns']['perfect_match'] == 1:
                        experiment_labels[exp_label][difficulty]['columns'].append(0)  # Perfect match
                    else:
                        experiment_labels[exp_label][difficulty]['columns'].append(1)  # Semantic error
        
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    
    difficulties = sorted(set(
        difficulty for data in experiment_labels.values() for difficulty in data.keys()
    ), key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)

    print(difficulties)
    
    if len(difficulties) == 1:
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        axs = [ax]
    else:
        fig, axs = plt.subplots(2, len(difficulties), figsize=(6 * len(difficulties), 8), sharey=True, sharex=True)
        if len(difficulties) == 2:
            axs = list(axs)  # Ensure axs is always a list
    
    # Define error type labels for better visualization
    error_labels = {0: 'Perfect Match', 1: 'Semantic Error', 2: 'Execution Error'}
    
    for i, difficulty in enumerate(difficulties):
        for metric_idx, metric_type in enumerate(['oids', 'columns']):
            # Combine data for all experiments for this metric type
            combined_data = []
            
            for exp_label, data in experiment_labels.items():
                if difficulty in data and data[difficulty]:  # Only process if there are errors
                    error_list = data[difficulty][metric_type]
                    error_counts = pd.Series(error_list).value_counts().reset_index()
                    error_counts.columns = ['error_type', 'error_count']
                    error_counts['experiment'] = exp_label
                    
                    if percentage:
                        total_count = total_queries_per_exp_diff[exp_label][difficulty]
                        error_counts['error_count'] = (error_counts['error_count'] / total_count) * 100
                    
                    combined_data.append(error_counts)
            
            if not combined_data:
                print(f"No data for difficulty {difficulty} and metric {metric_type} in model {model_name}.")
                axs[metric_idx, i].text(0.5, 0.5, 'No Data', horizontalalignment='center', 
                           verticalalignment='center', transform=axs[metric_idx, i].transAxes, fontsize=16)
                axs[metric_idx, i].set_xlabel(f"{difficulty.replace('advanced','hard').title()}", fontsize=16)
                continue
            error_data = pd.concat(combined_data, ignore_index=True)
            
            # Map error types to labels
            error_data['error_type'] = error_data['error_type'].map(error_labels)
            # Create the bar plot
            sns.barplot(x='error_type', y='error_count', data=error_data, 
                    ax=axs[metric_idx, i], hue='experiment')
            # Add value labels on top of each bar
            for p in axs[metric_idx, i].patches:
                height = p.get_height()
                if height > 0:
                    if percentage:
                        axs[metric_idx, i].annotate(f'{height:.1f}%',
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 3),
                                        textcoords='offset points',
                                        fontsize=10)
                    else:
                        axs[metric_idx, i].annotate(f'{int(height)}',
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 3),
                                        textcoords='offset points',
                                        fontsize=10)
            # Set titles and labels - put difficulty label at bottom as xlabel
            axs[metric_idx, i].set_xlabel(f"{difficulty.replace('advanced','hard').title()}", fontsize=20)
            if metric_idx == 0:
                axs[metric_idx, i].set_title("")  # Remove title from top
            if metric_idx == 0:
                if i == 0:  # Only show y-label on first subplot
                    axs[metric_idx, i].set_ylabel("Rows", fontsize=20)
            else:
                axs[metric_idx, i].set_ylabel("Columns", fontsize=20)
            # Rotate x-axis labels
            axs[metric_idx, i].tick_params(axis='x', labelsize=16)
            axs[metric_idx, i].tick_params(axis='y', labelsize=16)
            # Add legend only to the last subplot
            if metric_idx == 0 and i == len(difficulties) - 1:
                axs[metric_idx, i].legend(bbox_to_anchor=(0.65, 1), loc='upper left', fontsize=12)
            else:
                axs[metric_idx, i].get_legend().remove()
    # Adjust layout
    plt.subplots_adjust(hspace=0.045, wspace=0.025)
    # Set y-axis label
    if percentage:
        fig.supylabel('Percentage (%)', fontsize=24, x=0.055)

        # set limit to 100
        for ax in axs.flatten():
            ax.set_ylim(0, 100)
    else:
        plt.ylabel("Count", fontsize=24)
    # set x-axis labels fontsize
    for ax in axs.flatten():
        ax.tick_params(axis='x', labelsize=13)
    # Set main title
    if 'claude-3-7' in model_name: model_label = 'Claude-3.7'
    elif 'gpt' in model_name: 
        model_label = "-".join(model_name.split('-')[0:2]).replace('gpt', 'GPT')
    else: model_label = model_name
    plt.suptitle(f"Query Results Distribution by Generation Method - {model_label}", fontsize=24, y=0.92)
    
    for ax in axs.flatten():
        ax.grid(True)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()



def plot_execution_errors_by_model_interactive(
        evaluation_results_: Dict[str, Dict[str, Any]],
) -> None:


    import copy
    evaluation_results = copy.deepcopy(evaluation_results_)
    # Extract model names from the results dictionary
    models = sorted(list(evaluation_results.keys()))
    
    # Create widgets for interactive selection
    model_dropdown = widgets.Dropdown(
        options=models,
        description='Model:',
        value=models[0] if models else None,
        layout=widgets.Layout(width='50%')
    )
    
    y_lim = widgets.Dropdown(
        options=[('Count', False), ('Percentage', True)],
        description='Y-axis:',
        value=True,
        layout=widgets.Layout(width='50%')
    )

    error_type = widgets.Dropdown(
        options=[('All',False), ('Type', True)],
        description='Error Type:',
        value=False,
        layout=widgets.Layout(width='50%')
    )

    med_hard_dropdown = widgets.Dropdown(
        options=[('Simple-Medium-Hard', False), ('Simple-Medium/Hard', True)],
        description='Difficulties:',
        value=False,
        layout=widgets.Layout(width='50%')
    )
    
    output_widget = widgets.Output()

    def update_plot(model, y_lim, error_type, medium_hard):
        clear_output(wait=True)
        # experiments = sorted([exp for exp in evaluation_results[model].keys() if exp in exp_names])
        experiments = sorted(list(evaluation_results[model].keys()))

        experiment_labels = {}
    
        for self_corr_key in [ 'corrected', 'self_corrected',]:
            experiment_labels[self_corr_key] = {}
            for experiment in experiments:
            # Check if the experiment is self-corrected or corrected
            # Iterate through self-corrected and corrected results
                try:
                    if "dir" in experiment:
                        exp_label = "Direct"
                    if "sbs" in experiment:
                        exp_label = "Step-by-Step"
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    if medium_hard:
                        # Change medium and advanced difficulties to medium-hard
                        for res_i in results:
                            if res_i['difficulty'] == 'medium' or res_i['difficulty'] == 'advanced': res_i['difficulty'] = 'medium-hard'
                    error_types = []
                    total_exps = len(results)
                    print(total_exps)
                
                    for res in results:
                        error_info = res['comparison'].get('error_pred', None)
                        if error_info:
                            if error_type:
                                error_types.append(get_error_class(error_info).get('error_type').upper())
                            else:
                                error_types.append(get_error_class(error_info).get('error_class'))

                        else:
                            continue
                    
                    # Store the error types for each experiment
                    experiment_labels[self_corr_key][exp_label] = error_types
                    print(experiment_labels[self_corr_key][exp_label])
                
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment}: {e}")

        fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharex=True, sharey=True)
        for i, (self_corr_key, data) in enumerate(experiment_labels.items()):
            # group the error types by their class and by experiment, to plot them without stacking
            for j, (exp_label, error_list) in enumerate(data.items()):
                error_counts = pd.Series(error_list).value_counts().reset_index()
                # Change "QueryCanceled" to "Timeout"
                if error_type == False:
                    error_counts['index'] = error_counts['index'].replace({
                        'QueryCanceled': 'Timeout',})
                error_counts.columns = ['error_type', 'error_count']
                error_counts['experiment'] = exp_label
                if j == 0:
                    error_data = error_counts
                else:
                    error_data = pd.concat([error_data, error_counts], ignore_index=True)
                    
            if y_lim:
                # Convert error counts to percentages
                # total_count = error_data['error_count'].sum()
                total_count = total_exps
                if total_count > 0:
                    error_data['error_count'] = error_data['error_count'] / total_count * 100
                else:
                    error_data['error_count'] = 0
                sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
            else:
                # Plot error counts directly
                # error_data = error_data.groupby(['error_type', 'experiment']).sum().reset_index()
                # sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment', palette='Set2')
                sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
            plt.xticks(rotation=45, ha='right', fontsize=14)
            plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
        axs[0].set_xlabel("Without Self-Correction", fontsize=18)
        # axs[0].set_ylabel("Error Count", fontsize=18)
        axs[1].set_xlabel("With Self-Correction", fontsize=18)
        axs[0].yaxis.set_visible(True)
        axs[1].yaxis.set_visible(False)
        # if y_lim true, set y-axis limit to [0, 1]
        # if y_lim:
        # else:
        if y_lim:
            axs[0].set_ylim(0, 100)  # Set y-axis limit to [0, 100] for percentage
            axs[1].set_ylim(0, 100)  # Set y-axis limit to [0, 100] for percentage
            # set y-axis separately by 10
            axs[0].set_yticks(np.arange(0, 101, 10))
            axs[1].set_yticks(np.arange(0, 101, 10))
            axs[0].set_ylabel("Error Percentage of Total Queries", fontsize=18)
            
        else:
            axs[0].set_ylabel("Error Count", fontsize=18)


        plt.subplots_adjust(hspace=0.02, )
        plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
        
        plt.suptitle(f"Execution Errors, Direct vs Step-By-Step", fontsize=20)
        plt.tight_layout()
        plt.show()

    # Create interactive widget
    interact_widget = widgets.interactive(
        update_plot,
        model=model_dropdown,
        y_lim=y_lim,
        error_type=error_type,
        medium_hard=med_hard_dropdown
    )
    
    # Display the widgets and output
    display(widgets.VBox([interact_widget, output_widget]))
    
    # Initial plot
    if models:
        update_plot(models[0], True, False, False)




# def statistical_test(
#     evaluation_results: Dict[str, Dict[str, Any]],
#     model_names: List[str],
#     exp_names: List[str],
#     use_self_correction: bool = True,
#     formatted_columns: bool = True,
#     take_error_golds: bool = False,
#     table_size: str = 'small',
#     adjust_margins: bool = True,
# ) -> str:
#     """
#     Computes statistical significance tests (p-values) between different experiments
#     for specified models and generates LaTeX tables summarizing the results.
    
#     Args:
#         evaluation_results: Nested dict of evaluation results.
#         model_names: List of models to include.
#         exp_names: List of experiment names to rank.
#         use_self_correction: (bool) If True, uses "W/ Self-Corr" data. 
#                              If False, uses "W/o Self-Corr" data.
#         formatted_columns: Whether to use 'columns_formatted' metrics.
#         take_error_golds: Whether to include gold queries with errors.
#         table_size: LaTeX font size command.
#         adjust_margins: Whether to use 'table*' and squeeze columns.

#     Returns:
#         str: A single string containing the LaTeX for ALL ranking tables,
#              one after the other, separated by a '\\bigskip'.
#     """
#     from mlxtend.evaluate import permutation_test

#     # Attempt to import the aggregation utility
#     try:
#         from llm.utils.eval_utils import metrics_aggregation
#     except ImportError:
#         print("Warning: 'llm.utils.eval_utils.metrics_aggregation' not found.")
#         print("Please ensure this utility is available in your environment.")
#         def metrics_aggregation(results, take_error_gold):
#             print("Using placeholder 'metrics_aggregation' due to import error.")
#             if results:
#                 return results[0].get('summary', {})
#             return {}

    
#     # --- 1. Data Collection ---
#     # We only store the raw *mean* values for ranking.
#     # Key: (Metric_Type, Exp_Name, Model, Difficulty)
#     data_map = {} 
#     difficulties_found = set()
#     sorted_exp_names = sorted(exp_names)
#     sorted_models = sorted(model_names)
    
#     sc_label_key = "W/ Self-Corr" if use_self_correction else "W/o Self-Corr"
#     sc_title = "W/ Self-Correction" if use_self_correction else "W/o Self-Correction"
#     self_corr_key = 'self_corrected' if use_self_correction else 'corrected'
    
#     for model in sorted_models:

        
#         for exp_name in sorted_exp_names:
#             if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
#                 continue 
                
#             try:
#                 results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
#             except KeyError:
#                 continue
            
#             if not results_list:
#                 continue
                
#             agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
            
#             # We only need 'by_difficulty' for the mean scores
#             results_by_diff = agg_metrics.get('by_difficulty_runs', {}) 


#             for diff, metrics in results_by_diff.items():
#                 diff_clean = diff.replace('advanced', 'hard').capitalize()
#                 difficulties_found.add(diff_clean)
#                 for run_id, exp_value in results_by_diff.get(diff, {}).get('runs_metrics', {}).items():
#                     oid_value = exp_value.get('oids', {}).get('perfect_match_rate', 0)
                
#                     col_metrics = exp_value.get('columns_formatted', metrics.get('columns', {})) if formatted_columns else metrics.get('columns', {})
#                     col_value = col_metrics.get('perfect_match_rate', 0)

#                     # Store the raw float values, not strings
#                     data_map[('ID Match', exp_name, model, diff_clean, run_id)] = oid_value
#                     data_map[('Column Match', exp_name, model, diff_clean, run_id)] = col_value

#                 # ...

#         # calculate p-value between experiments for each metric and difficulty
#         # p_value = permutation_test(direct_exp, step_by_step_exp,
#         #                         method='approximate',
#         #                         num_rounds=10000,
#         #                         seed=0)

                
#     # --- 2. Build and Rank Tables per Experiment ---
    
#     diff_order = ['Simple', 'Medium', 'Hard']
#     sorted_diffs = sorted(list(difficulties_found), 
#                           key=lambda x: diff_order.index(x) if x in diff_order else 99)
#     metric_labels = ['ID Match', 'Column Match']


def statistical_test(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    use_self_correction: bool = True,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
) -> str:
    """
    Computes statistical significance tests (permutation test) between 
    two experiments (e.g., Direct vs Step-by-Step) for specified models 
    and generates a LaTeX table (in 'Wide/Parallel' format) of p-values.
    
    Bold values indicate statistical significance (p < 0.05).
    """
    
    try:
        from mlxtend.evaluate import permutation_test
    except ImportError:
        print("Warning: 'mlxtend' library not found.")
        return "% Error: mlxtend library not found. Please install it to use this function."

    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        def metrics_aggregation(results, take_error_gold):
            if results: return results[0].get('summary', {})
            return {}

    # --- 1. Data Collection ---
    # We need lists of run-values to perform the permutation test.
    # samples[model][metric][difficulty][exp_name] = [val1, val2, ...]
    samples = {model: {'ID Match': {}, 'Column Match': {}} for model in model_names}
    
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    # Ensure we have at least two experiments to compare
    if len(sorted_exp_names) < 2:
        return "% Error: Need at least 2 experiments in 'exp_names' to perform a comparison."
    
    # We will compare exp_names[0] vs exp_names[1]
    exp_A = sorted_exp_names[0]
    exp_B = sorted_exp_names[1]
    
    sc_title = "W/ Self-Correction" if use_self_correction else "W/o Self-Correction"
    self_corr_key = 'self_corrected' if use_self_correction else 'corrected'
    
    for model in sorted_models:
        for exp_name in [exp_A, exp_B]:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue 
                
            try:
                results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
            except KeyError:
                continue
            
            if not results_list:
                continue
                
            agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
            
            # Get RUN-LEVEL metrics
            results_by_diff = agg_metrics.get('by_difficulty_runs', {}) 

            for diff, metrics in results_by_diff.items():
                diff_clean = diff.replace('advanced', 'hard').capitalize()
                difficulties_found.add(diff_clean)
                
                # Initialize lists for this difficulty if not present
                if diff_clean not in samples[model]['ID Match']:
                    samples[model]['ID Match'][diff_clean] = {exp_A: [], exp_B: []}
                    samples[model]['Column Match'][diff_clean] = {exp_A: [], exp_B: []}
                
                # Iterate over individual runs to get the distribution
                runs_data = metrics.get('runs_metrics', {})
                for run_id, run_vals in runs_data.items():
                    # OID Value
                    oid_val = run_vals.get('oids', {}).get('perfect_match_rate', 0)
                    samples[model]['ID Match'][diff_clean][exp_name].append(oid_val)
                    
                    # Column Value
                    col_metrics = run_vals.get('columns_formatted', run_vals.get('columns', {})) if formatted_columns else run_vals.get('columns', {})
                    col_val = col_metrics.get('perfect_match_rate', 0)
                    samples[model]['Column Match'][diff_clean][exp_name].append(col_val)

    # --- 2. Perform Permutation Tests ---
    # p_values[model][metric][difficulty] = p_val (float)
    p_values = {}
    
    for model in sorted_models:
        p_values[model] = {}
        for metric in ['ID Match', 'Column Match']:
            p_values[model][metric] = {}
            for diff in difficulties_found:
                # Get samples for both experiments
                try:
                    dist_A = samples[model][metric][diff][exp_A]
                    dist_B = samples[model][metric][diff][exp_B]
                except KeyError:
                    p_values[model][metric][diff] = None
                    continue
                
                if not dist_A or not dist_B:
                    p_values[model][metric][diff] = None
                    continue
                    
                # Run Test
                # approx method is faster for large number of rounds
                p_val = permutation_test(
                    dist_A, dist_B,
                    method='approximate',
                    num_rounds=100000,
                    seed=341987
                )
                p_values[model][metric][diff] = p_val
    print(p_values)
    # --- 3. Build LaTeX Table (Parallel Format) ---
    
    # Add table size and centering
    size_commands = {
        'tiny': '\\tiny',
        'scriptsize': '\\scriptsize', 
        'footnotesize': '\\footnotesize',
        'small': '\\small',
        'normalsize': '\\normalsize',
        'large': '\\large'
    }

    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)
    
    latex_lines = []
    table_env = "table*" if adjust_margins else "table"
    size_cmd = size_commands.get(table_size, '\\small')
    
    latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
    latex_lines.append(size_cmd)
    latex_lines.append(r"\centering")
    if adjust_margins:
        latex_lines.append(r"\setlength{\tabcolsep}{4pt}") 

    # Wide format layout
    num_diffs = len(sorted_diffs)
    num_data_cols = num_diffs * 2 
    total_cols = 1 + num_data_cols
    
    col_def = "l | " + ("c" * num_diffs) + " | " + ("c" * num_diffs)
    latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
    latex_lines.append(r"\toprule")

    # -- Header 1 --
    header_1 = [""]
    header_1.append(f"\\multicolumn{{{num_diffs}}}{{c|}}{{ID Match (p-values)}}")
    header_1.append(f"\\multicolumn{{{num_diffs}}}{{c|}}{{Column Match (p-values)}}")
    latex_lines.append(" & ".join(header_1) + r" \\")

    # -- Header 2 --
    header_2 = ["Model"] + sorted_diffs + sorted_diffs
    latex_lines.append(" & ".join(header_2) + r" \\")
    latex_lines.append(r"\midrule")

    # -- Data Rows --
    for model in sorted_models:
        if 'claude-3-7' in model: model_label = 'Claude-3.7'
        elif 'gpt' in model: 
            model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
        else: model_label = model
        
        line = [model_label]
        
        # ID Match Block
        for diff in sorted_diffs:
            p = p_values[model]['ID Match'].get(diff, None)
            if p is None:
                line.append("-")
            else:
                val_str = f"{p:.3f}"
                if p < 0.05: # Significant
                    line.append(f"\\textbf{{{val_str}}}")
                else:
                    line.append(val_str)
                    
        # Column Match Block
        for diff in sorted_diffs:
            p = p_values[model]['Column Match'].get(diff, None)
            if p is None:
                line.append("-")
            else:
                val_str = f"{p:.3f}"
                if p < 0.05: # Significant
                    line.append(f"\\textbf{{{val_str}}}")
                else:
                    line.append(val_str)
        
        latex_lines.append(" & ".join(line) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    exp_A_display = 'direct' if 'dir' in exp_A else 'step-by-step' if 'sbs' in exp_A else exp_A
    exp_B_display = 'direct' if 'dir' in exp_B else 'step-by-step' if 'sbs' in exp_B else exp_B

    caption = (
        f"Statistical Significance (p-values) for {exp_A_display} vs {exp_B_display} "
        f"({sc_title}). Values $< 0.05$ are bolded."
    )
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(r"\label{tab:significance_test}")
    latex_lines.append(f"\\end{{{table_env}}}")
    
    final_str = "\n".join(latex_lines)
    
    print("\n" + "="*80)
    print(f"Statistical Significance Table ({sc_title})")
    print("="*80)
    print(final_str)
    print("="*80)
    
    return final_str
    



def generate_significance_ranking_table(
    evaluation_results: Dict[str, Dict[str, Any]],
    model_names: List[str],
    exp_names: List[str],
    use_self_correction: bool = True,
    formatted_columns: bool = True,
    take_error_golds: bool = False,
    table_size: str = 'small',
    adjust_margins: bool = True,
) -> str:
    """
    Creates a Ranking Table where ranks are determined by statistical significance.
    
    Algorithm:
    1. For each column (Exp, Metric, Diff), models are sorted by score.
    2. Model 1 is Rank 1.
    3. Model N is compared to Model N-1 using permutation test.
       - If NOT Significant (p >= 0.05): Model N gets same Rank as Model N-1.
       - If Significant (p < 0.05): Model N gets Rank = N (standard ranking).
       
    Output format matches generate_latex_ranking_tables.
    """
    try:
        from mlxtend.evaluate import permutation_test
    except ImportError:
        print("Warning: 'mlxtend' library not found.")
        return "% Error: mlxtend library not found."

    # Attempt to import the aggregation utility
    try:
        from llm.utils.eval_utils import metrics_aggregation
    except ImportError:
        def metrics_aggregation(results, take_error_gold):
            if results: return results[0].get('summary', {})
            return {}

    # --- 1. Data Collection ---
    # Need run-level data AND mean scores
    # Structure: data[exp][metric][diff] = list of (model_name, mean_score, [run_values])
    
    data_store = {} 
    
    difficulties_found = set()
    sorted_exp_names = sorted(exp_names)
    sorted_models = sorted(model_names)
    
    sc_title = "W/ Self-Correction" if use_self_correction else "W/o Self-Correction"
    self_corr_key = 'self_corrected' if use_self_correction else 'corrected'
    
    # Initialize data structure
    for exp in sorted_exp_names:
        data_store[exp] = {'ID Match': {}, 'Column Match': {}}
    
    for exp_name in sorted_exp_names:
        for model in sorted_models:
            if model not in evaluation_results or exp_name not in evaluation_results.get(model, {}):
                continue 
            try:
                results_list = evaluation_results[model][exp_name][self_corr_key]['detailed_results']
            except KeyError:
                continue
            if not results_list: continue
                
            agg_metrics = metrics_aggregation(results=results_list, take_error_gold=take_error_golds)
            results_by_diff = agg_metrics.get('by_difficulty_runs', {}) 

            for diff, metrics in results_by_diff.items():
                diff_clean = diff.replace('advanced', 'hard').capitalize()
                difficulties_found.add(diff_clean)
                
                # Initialize list if needed
                if diff_clean not in data_store[exp_name]['ID Match']:
                    data_store[exp_name]['ID Match'][diff_clean] = []
                    data_store[exp_name]['Column Match'][diff_clean] = []
                
                # Collect Runs
                oid_runs = []
                col_runs = []
                runs_data = metrics.get('runs_metrics', {})
                for _, run_vals in runs_data.items():
                    oid_runs.append(run_vals.get('oids', {}).get('perfect_match_rate', 0))
                    col_m = run_vals.get('columns_formatted', run_vals.get('columns', {})) if formatted_columns else run_vals.get('columns', {})
                    col_runs.append(col_m.get('perfect_match_rate', 0))
                
                # Calculate Mean (or use pre-calc)
                oid_mean = sum(oid_runs)/len(oid_runs) if oid_runs else 0
                col_mean = sum(col_runs)/len(col_runs) if col_runs else 0
                
                data_store[exp_name]['ID Match'][diff_clean].append({
                    'model': model,
                    'mean': oid_mean,
                    'runs': oid_runs
                })
                data_store[exp_name]['Column Match'][diff_clean].append({
                    'model': model,
                    'mean': col_mean,
                    'runs': col_runs
                })

    # --- 2. Compute Ranks with Significance ---
    # ranks_map[exp][model] = {'RS': 1, 'RM': 2...}
    ranks_map = {exp: {m: {} for m in sorted_models} for exp in sorted_exp_names}
    
    diff_order = ['Simple', 'Medium', 'Hard']
    sorted_diffs = sorted(list(difficulties_found), 
                          key=lambda x: diff_order.index(x) if x in diff_order else 99)

    for exp_name in sorted_exp_names:
        for metric in ['ID Match', 'Column Match']:
            for diff in sorted_diffs:
                # Get list of models for this specific column
                entries = data_store[exp_name][metric].get(diff, [])
                
                if not entries: continue
                
                # 1. Sort descending by mean score
                entries.sort(key=lambda x: x['mean'], reverse=True)
                
                # 2. Assign Ranks
                # First model is Rank 1
                current_entries_ranks = {} # model -> rank
                
                # Base case
                current_rank = 1
                current_entries_ranks[entries[0]['model']] = 1
                
                # Iterate rest
                for i in range(1, len(entries)):
                    prev_model = entries[i-1]
                    curr_model = entries[i]
                    
                    # Permutation Test
                    if not prev_model['runs'] or not curr_model['runs']:
                         # Fallback if no runs: Treat as tied? Or distinct?
                         # Let's treat as distinct rank to be safe
                         current_entries_ranks[curr_model['model']] = i + 1
                         continue

                    p_val = permutation_test(
                        prev_model['runs'], 
                        curr_model['runs'],
                        method='approximate',
                        num_rounds=5000, # Lower rounds for speed in loop
                        seed=0
                    )
                    
                    if p_val < 0.05:
                        # Significant difference: Standard Rank (Index + 1)
                        # e.g. if i=1 (2nd person), rank is 2
                        current_rank = i + 1
                    else:
                        # Not significant: Keep previous rank
                        # current_rank stays same
                        pass
                    
                    current_entries_ranks[curr_model['model']] = current_rank
                
                # Store in main map
                short_metric = 'R' if metric == 'ID Match' else 'C'
                col_key = f"{short_metric}{diff[0]}" # e.g. RS, CM
                
                for model_name, rank in current_entries_ranks.items():
                    ranks_map[exp_name][model_name][col_key] = rank

    # --- 3. Generate Tables ---
    # Add table size and centering
    size_commands = {
        'tiny': '\\tiny',
        'scriptsize': '\\scriptsize', 
        'footnotesize': '\\footnotesize',
        'small': '\\small',
        'normalsize': '\\normalsize',
        'large': '\\large'
    }
    
    all_latex_tables = []
    rank_col_headers = []
    for m_short in ['R', 'C']:
        for d_short in [d[0] for d in sorted_diffs]:
            rank_col_headers.append(f"{m_short}{d_short}")

    for exp_name in sorted_exp_names:
        
        # Build DataFrame for sorting and printing
        rows = []
        for model in sorted_models:
            # Display Name
            if 'claude-3-7' in model: model_label = 'Claude-3.7'
            elif 'gpt' in model: model_label = "-".join(model.split('-')[0:2]).replace('gpt', 'GPT')
            else: model_label = model
            
            r_data = {'Model': model_label}
            rank_sum = 0
            has_data = False
            
            # Fetch ranks
            for col_key in rank_col_headers:
                r = ranks_map[exp_name][model].get(col_key, -1)
                if r != -1:
                    r_data[col_key] = r
                    rank_sum += r
                    has_data = True
                else:
                    r_data[col_key] = "-"
            
            if has_data:
                r_data['SUM'] = rank_sum
                rows.append(r_data)
        
        if not rows: continue
            
        # Sort by SUM
        rows.sort(key=lambda x: x['SUM'])
        
        # LaTeX Construction
        latex_lines = []
        table_env = "table*" if adjust_margins else "table"
        latex_lines.append(f"\\begin{{{table_env}}}[htbp]")
        latex_lines.append(size_commands.get(table_size, '\\small'))
        latex_lines.append(r"\centering")

        col_def = "l | " + " ".join(["c"] * len(sorted_diffs)) + " | " \
                  + " ".join(["c"] * len(sorted_diffs)) + " || c"
        
        latex_lines.append(f"\\begin{{tabular}}{{{col_def}}}")
        latex_lines.append(r"\toprule")
        
        header = ["LLM"] + rank_col_headers + ["SUM"]
        latex_lines.append(" & ".join(header) + r" \\")
        latex_lines.append(r"\midrule")
        
        for row in rows:
            vals = [row['Model']]
            for k in rank_col_headers:
                vals.append(str(row[k]))
            vals.append(str(row['SUM']))
            latex_lines.append(" & ".join(vals) + r" \\")

        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        
        exp_display = 'Step-by-Step' if 'sbs' in exp_name else exp_name
        exp_display = 'Direct' if 'direct' in exp_name else exp_display
        caption = f"Statistical Ranking for {exp_display} ({sc_title}). Models with no significant difference ($p \ge 0.05$) share the same rank."
        latex_lines.append(f"\\caption{{{caption}}}")
        
        label_exp = "_sbs" if 'sbs' in exp_name else "_direct"
        label_sc = "_w_sc" if use_self_correction else "_wo_sc"
        latex_lines.append(f"\\label{{tab:sig_rank_{label_exp}{label_sc}}}")
        latex_lines.append(f"\\end{{{table_env}}}")
        
        all_latex_tables.append("\n".join(latex_lines))

    final_str = "\n\n\\bigskip\n\n".join(all_latex_tables)
    print("\n" + "="*80)
    print(f"Significance Ranking Tables ({sc_title})")
    print("="*80)
    print(final_str)
    print("="*80)
    
    return final_str


def plot_diff_classification(evaluation_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Plots difficulty classification metrics (accuracy, precision, recall, F1)
    for different model-experiment combinations.

    Args:
        evaluation_results: A nested dictionary where:
            - keys are model names,
            - values are dicts with experiment names as keys,
            - each experiment contains a 'corrected' key with detailed predictions.

    Note:
        Only experiments using sql_gen_method != 'direct' are included.
    """
    self_corr_key = 'corrected'  # expected key for self-corrected outputs

    rows = []
    for model_name, model_experiments in evaluation_results.items():
        for experiment_name, experiment_data in model_experiments.items():
            try:
                exp_block = experiment_data[self_corr_key]

                # Skip if method is direct
                if exp_block['metadata']['sql_gen_method'] == 'direct':
                    continue

                detailed_results = exp_block['detailed_results']
                pred_list = [entry['pred_diff'] for entry in detailed_results]
                true_list = [entry['difficulty'] for entry in detailed_results]

                # Compute metrics
                accuracy = accuracy_score(true_list, pred_list)
                precision = precision_score(true_list, pred_list, average='weighted', zero_division=0)
                recall = recall_score(true_list, pred_list, average='weighted', zero_division=0)
                f1 = f1_score(true_list, pred_list, average='weighted', zero_division=0)

                # Short label for plotting
                short_label = f"{model_name.split('-202')[0]}_{experiment_name.replace('alerce_', '').replace('_', '')}"

                rows.append((short_label, 'Accuracy', accuracy))
                rows.append((short_label, 'Precision', precision))
                rows.append((short_label, 'Recall', recall))
                rows.append((short_label, 'F1 Score', f1))

            except KeyError as e:
                print(f"Warning: Skipping {model_name} - {experiment_name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["Experiment", "Metric", "Score"])

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    ax = sns.barplot(data=df, x="Experiment", y="Score", hue="Metric", palette="Set2")

    # Add score labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=2)

    ax.set_ylim(0, 1)
    ax.set_title("Difficulty Classification Metrics by Model and Experiment", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xlabel("Experiment")
    plt.xticks(rotation=15, ha='center')
    ax.legend(title="Metric", loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_diff_classification_conf_matrix(evaluation_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Interactive confusion matrix plot for difficulty classification.

    Allows users to select model, experiment, and normalization option.

    Args:
        evaluation_results: A nested dictionary with model -> experiment -> evaluation data.

    Returns:
        None. Displays interactive confusion matrix plot.
    """
    self_corr_key = 'corrected'

    # Build a nested structure of valid (non-direct) experiments
    valid_data = {}
    for model, experiments in evaluation_results.items():
        for experiment, data in experiments.items():
            try:
                if data[self_corr_key]['metadata']['sql_gen_method'] == 'direct':
                    continue
                if model not in valid_data:
                    valid_data[model] = {}
                valid_data[model][experiment] = data[self_corr_key]['detailed_results']
            except KeyError:
                warnings.warn(f"Skipping {model} - {experiment} due to missing data.")

    # Widgets
    model_selector = widgets.Dropdown(
        options=sorted(valid_data.keys()),
        description='Model:',
        layout=widgets.Layout(width='50%')
    )

    experiment_selector = widgets.Dropdown(
        options=[],
        description='Experiment:',
        layout=widgets.Layout(width='50%')
    )

    normalize_toggle = widgets.Checkbox(
        value=False,
        description='Normalize',
        indent=False
    )

    # Update experiments based on model selection
    def update_experiments(model_name):
        experiment_selector.options = sorted(valid_data[model_name].keys())

    model_selector.observe(lambda change: update_experiments(change['new']), names='value')

    # Function to plot confusion matrix
    def plot_conf_matrix(model_name, experiment_name, normalize):
        detailed_results = valid_data[model_name][experiment_name]
        true_labels = [res['difficulty'] for res in detailed_results]
        pred_labels = [res['pred_diff'] for res in detailed_results]
        all_labels = sorted(set(true_labels + pred_labels), reverse=True)

        cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
            title_suffix = ' (Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=all_labels, yticklabels=all_labels)
        plt.title(f'Confusion Matrix: {model_name} / {experiment_name}{title_suffix}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()

    # Main interactive display
    def interactive_plot(model_name, experiment_name, normalize):
        plot_conf_matrix(model_name, experiment_name, normalize)

    # Initialize experiment dropdown
    update_experiments(model_selector.value)

    interact(interactive_plot,
             model_name=model_selector,
             experiment_name=experiment_selector,
             normalize=normalize_toggle)

def compute_schema_linking_metrics_by_experiment(
    evaluation_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Computes schema linking metrics (Accuracy, Precision, Recall, F1) per experiment for each model.
    Reports both macro- and micro-averaged metrics within each experiment.

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): Nested dictionary of model → experiment → results.

    Returns:
        None. Prints metrics per model and experiment.
    """

    def compute_pairwise_metrics(pred: List[str], gold: List[str]) -> Tuple[float, float, float, float, int, int, int, bool]:
        """Returns: accuracy, precision, recall, f1, TP, FP, FN, exact_match"""
        pred_set = set(pred)
        gold_set = set(gold)
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = 1.0 if pred_set == gold_set else 0.0
        exact = pred_set == gold_set
        return accuracy, precision, recall, f1, tp, fp, fn, exact

    for model, experiments in evaluation_results.items():
        print(f"\n========== Model: {model} ==========")
        for experiment, content in experiments.items():
            try:
                detailed = content['corrected']['detailed_results']
                macro_acc, macro_prec, macro_rec, macro_f1 = [], [], [], []
                total_tp = total_fp = total_fn = 0
                exact_matches = 0

                for item in detailed:
                    pred = item.get('pred_tables', [])
                    gold = item.get('gold_tables', [])
                    # transform string list to list of strings
                    import ast
                    if isinstance(pred, str): pred = ast.literal_eval(pred)
                    if isinstance(gold, str): gold = ast.literal_eval(gold)
                    if gold == []:
                        # If no gold tables, skip this item
                        continue
                    acc, prec, rec, f1, tp, fp, fn, exact = compute_pairwise_metrics(pred, gold)
                    macro_acc.append(acc)
                    macro_prec.append(prec)
                    macro_rec.append(rec)
                    macro_f1.append(f1)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    if exact:
                        exact_matches += 1

                num_examples = len(detailed)
                print(f"\n--- Experiment: {experiment} ---")
                print("Macro-Averaged (example-level):")
                print(f"  Accuracy:  {np.mean(macro_acc):.3f}")
                print(f"  Precision: {np.mean(macro_prec):.3f}")
                print(f"  Recall:    {np.mean(macro_rec):.3f}")
                print(f"  F1-score:  {np.mean(macro_f1):.3f}")
                print(f"  Exact Match Rate: {exact_matches}/{num_examples} = {exact_matches / num_examples:.3f}")

                print("Micro-Averaged (token-level):")
                micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
                print(f"  Precision: {micro_prec:.3f}")
                print(f"  Recall:    {micro_rec:.3f}")
                print(f"  F1-score:  {micro_f1:.3f}")

            except KeyError as e:
                print(f"Skipping {model} - {experiment}: {e}")

def compute_schema_linking_metrics_interactive(
    evaluation_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Computes schema linking metrics per experiment with interactive model/experiment selection.
    Reports macro- and micro-averaged metrics per experiment (no cross-experiment aggregation).

    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): Nested dictionary of model → experiment → results.
    """

    def compute_pairwise_metrics(pred: List[str], gold: List[str]) -> Tuple[float, float, float, float, int, int, int, bool]:
        pred_set = set(pred)
        gold_set = set(gold)
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = 1.0 if pred_set == gold_set else 0.0
        exact = pred_set == gold_set
        return accuracy, precision, recall, f1, tp, fp, fn, exact

    # Widget options
    model_names = sorted(evaluation_results.keys())
    experiment_names = sorted({
        exp
        for model in model_names
        for exp in evaluation_results[model].keys()
    })

    model_dropdown = widgets.Dropdown(options=["All"] + model_names, description="Model:")
    experiment_dropdown = widgets.Dropdown(options=["All"] + experiment_names, description="Experiment:")
    output = widgets.Output()

    def on_selection_change(change=None):
        with output:
            clear_output()
            selected_model = model_dropdown.value
            selected_experiment = experiment_dropdown.value

            for model, experiments in evaluation_results.items():
                if selected_model != "All" and model != selected_model:
                    continue

                for experiment, content in experiments.items():
                    if selected_experiment != "All" and experiment != selected_experiment:
                        continue

                    try:
                        detailed = content['corrected']['detailed_results']
                        macro_acc, macro_prec, macro_rec, macro_f1 = [], [], [], []
                        total_tp = total_fp = total_fn = 0
                        exact_matches = 0

                        for item in detailed:
                            pred = item.get('pred_tables', [])
                            gold = item.get('gold_tables', [])
                            # transform string list to list of strings
                            import ast
                            if isinstance(pred, str): pred = ast.literal_eval(pred)
                            if isinstance(gold, str): gold = ast.literal_eval(gold)
                            if gold == []: continue # If no gold tables, skip this item
                            acc, prec, rec, f1, tp, fp, fn, exact = compute_pairwise_metrics(pred, gold)
                            macro_acc.append(acc)
                            macro_prec.append(prec)
                            macro_rec.append(rec)
                            macro_f1.append(f1)
                            total_tp += tp
                            total_fp += fp
                            total_fn += fn
                            if exact:
                                exact_matches += 1

                        num_examples = len(detailed)
                        print(f"\n=== Model: {model} | Experiment: {experiment} ===")
                        print("Macro-Averaged (example-level):")
                        print(f"  Accuracy:  {np.mean(macro_acc):.3f}")
                        print(f"  Precision: {np.mean(macro_prec):.3f}")
                        print(f"  Recall:    {np.mean(macro_rec):.3f}")
                        print(f"  F1-score:  {np.mean(macro_f1):.3f}")
                        print(f"  Exact Match Rate: {exact_matches}/{num_examples} = {exact_matches / num_examples:.3f}")

                        print("Micro-Averaged (token-level):")
                        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
                        print(f"  Precision: {micro_prec:.3f}")
                        print(f"  Recall:    {micro_rec:.3f}")
                        print(f"  F1-score:  {micro_f1:.3f}")

                    except KeyError as e:
                        print(f"Skipping {model} - {experiment}: {e}")

    # Connect interaction
    model_dropdown.observe(on_selection_change, names='value')
    experiment_dropdown.observe(on_selection_change, names='value')

    # Initial display
    display(widgets.HBox([model_dropdown, experiment_dropdown]))
    display(output)
    on_selection_change()













def plot_perfect_match_by_request_interactive(
        evaluation_results_: Dict[str, Dict[str, Any]],
        formatted_columns: bool = True,
        take_error_golds: bool = False,
) -> None:
    """
    Interactive plot of percentage of perfect match rates for OID and column matches by request and difficulty from evaluation results.
    It shows the percentage of perfect matches by each request separated in three subplots for simple, medium, and advanced difficulties.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): A dictionary containing evaluation results with keys
                                   for models and experiments, and values containing metrics.
        formatted_columns (bool): If True, uses formatted column results; otherwise uses regular column results.

    Returns:
        None: This function displays an interactive plot but does not return any value.
    """
    import copy
    evaluation_results = copy.deepcopy(evaluation_results_)
    # Extract model names from the results dictionary
    models = sorted(list(evaluation_results.keys()), reverse=True)
    
    # Create widgets for interactive selection
    model_dropdown = widgets.Dropdown(
        options=models,
        description='Model:',
        value=models[0] if models else None,
        layout=widgets.Layout(width='50%')
    )
    
    self_corr_dropdown = widgets.Dropdown(
        options=[('With Self-Correction', True), ('Without Self-Correction', False)],
        description='Self-Correction:',
        value=True,
        layout=widgets.Layout(width='50%')
    )

    med_hard_dropdown = widgets.Dropdown(
        options=[('Simple-Medium-Hard', False), ('Simple-Medium/Hard', True)],
        description='Difficulties:',
        value=False,
        layout=widgets.Layout(width='50%')
    )

    diff_plot_dropdown = widgets.Dropdown(
        options=[('All Difficulties', 'all'), ('Simple', 'simple'), ('Medium', 'medium'), ('Medium-Hard', 'medium-hard'), ('Advanced', 'advanced')],
        description='Plot Difficulty:',
        value='all',
        layout=widgets.Layout(width='50%')
    )
    
    output_widget = widgets.Output()
    
    from llm.utils.eval_utils import metrics_aggregation
    
    def update_plot(model, self_corr, medium_hard, diff_plot):
        """Update the plot based on selected model and self-correction setting"""
        with output_widget:
            clear_output(wait=True)
            evaluation_results = copy.deepcopy(evaluation_results_)
            
            # Self-corrected results key
            self_corr_key = 'self_corrected' if self_corr else 'corrected'

            # Medium-hard setting
            medium_hard = medium_hard
            
            # Initialize data collection lists
            experiment_labels = {}
            difficulties = []
            req_list = {}
            
            # Collect data for plotting
            experiments = sorted(list(evaluation_results[model].keys()))
            for experiment in experiments:
                try:
                    exp_label = f"{model}-{experiment}"
                    oid_match_rates = {}
                    column_match_rates = {}
                    
                    results = evaluation_results[model][experiment][self_corr_key]['detailed_results']
                    print(results)

                    
                    if medium_hard:
                        # Change medium and advanced req_list to medium-hard
                        for res_i in results:
                            if res_i['difficulty'] == 'medium' or res_i['difficulty'] == 'advanced': res_i['difficulty'] = 'medium-hard'
                
                    aggregate_metrics = metrics_aggregation(results=results, take_error_gold=take_error_golds)
                    
                    results = aggregate_metrics['by_difficulty_req_id']

                    print(f"Number of gold queries with errors for experiment {exp_label}: {aggregate_metrics['errors']['gold_errors']}")
                    
                    # Iterate through results to extract match rates by difficulty
                    for diff, diff_metrics in results.items():
                        if diff not in difficulties:
                            difficulties.append(diff)
                        oid_match_rates[diff] = {}
                        column_match_rates[diff] = {}
                        req_list[diff] = []

                        for req, metrics in diff_metrics['req_id_metrics'].items():
                            if req not in req_list[diff]:
                                req_list[diff].append(req)

                            # Extract OID metrics
                            oid_match_rates[diff][str(req)] = metrics['oids']['perfect_match_rate']
                            
                            # Extract column metrics
                            if formatted_columns and 'columns_formatted' in metrics:
                                column_match_rates[diff][str(req)] = metrics['columns_formatted']['perfect_match_rate']
                            else:
                                column_match_rates[diff][str(req)] = metrics['columns']['perfect_match_rate']

                    # Add metrics to experiment_labels
                    experiment_labels[exp_label] = {
                        'oid': oid_match_rates,
                        'column': column_match_rates
                    }
                except KeyError as e:
                    print(f"Warning: Missing data for {model} - {experiment}: {e}")

            # Sort req_list
            difficulties = sorted(difficulties, key=lambda x: ['simple', 'medium', 'medium-hard', 'advanced'].index(x))

            if diff_plot in difficulties:
                # Create the plot
                fig, axs = plt.subplots(2, 1, figsize=(20, 10))
                
                # Plot each experiment
                for i, (exp_label, data) in enumerate(experiment_labels.items()):
                    x = np.arange(len(req_list[diff_plot]))  # the label locations
                    width = 0.35  # the width of the bars
                    offset = (i - len(experiment_labels)/2) * width / len(experiment_labels)  # offset for multiple experiments
                    oid_rates = [data['oid'][diff_plot][str(req)] for req in sorted(req_list[diff_plot])]
                    column_rates = [data['column'][diff_plot][str(req)] for req in sorted(req_list[diff_plot])]

                    # Plot OID matches
                    bars1 = axs[0].bar(x + offset, oid_rates, width, label=exp_label)
                    
                    # Plot Column matches
                    bars2 = axs[1].bar(x + offset, column_rates, width, label=exp_label)
                    
                    # Add values on top of the bars
                    for bar in bars1:
                        height = bar.get_height()
                        axs[0].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

                    for bar in bars2:
                        height = bar.get_height()
                        axs[1].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

                    axs[0].set_xticks(x)
                    axs[0].set_xticklabels(sorted(req_list[diff_plot]))
                    axs[1].set_xticks(x)
                    axs[1].set_xticklabels(sorted(req_list[diff_plot]))
                    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
                    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
                # Set x-ticks and labels
                axs[0].set_ylabel('Rows', fontsize=20)
                axs[1].set_ylabel('Columns', fontsize=20)

            else:
                # Create the plot
                fig, axs = plt.subplots(2, len(difficulties), figsize=(40, 10))
                
                # Plot each experiment
                for i, (exp_label, data) in enumerate(experiment_labels.items()):
                    for j, diff in enumerate(difficulties):
                        x = np.arange(len(req_list[diff]))  # the label locations
                        width = 0.35  # the width of the bars
                        offset = (i - len(experiment_labels)/2) * width / len(experiment_labels)  # offset for multiple experiments
                        oid_rates = [data['oid'][diff][str(req)] for req in sorted(req_list[diff])]
                        column_rates = [data['column'][diff][str(req)] for req in sorted(req_list[diff])]

                        # Plot OID matches
                        bars1 = axs[0][j].bar(x + offset, oid_rates, width, label=exp_label)
                        
                        # Plot Column matches
                        bars2 = axs[1][j].bar(x + offset, column_rates, width, label=exp_label)
                        
                        # Add values on top of the bars
                        for bar in bars1:
                            height = bar.get_height()
                            axs[0][j].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

                        for bar in bars2:
                            height = bar.get_height()
                            axs[1][j].text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')

                        axs[0][j].set_xticks(x)
                        axs[0][j].set_xticklabels(sorted(req_list[diff]))
                        axs[1][j].set_xticks(x)
                        axs[1][j].set_xticklabels(sorted(req_list[diff]))
                        axs[0][j].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
                        axs[1][j].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
                # Set x-ticks and labels
                axs[0][0].set_ylabel('Rows', fontsize=20)
                axs[1][0].set_ylabel('Columns', fontsize=20)
                
            # change advanced to hard
            # req_list_labels = [d.replace('advanced', 'hard') for d in req_list]
            
            # Set title and labels
            plt.suptitle(f'Perfect Match Rates by Difficulty for {model}', fontsize=24)
            fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.02)
            
            # Add legend
            # plt.legend(title='Experiments', bbox_to_anchor=(1.02, 0.88))
            # plt.legend(title='Experiments', loc='upper left', bbox_to_anchor=(1.02, 1))
            # axs[0]
            plt.legend(title='Experiments', loc='upper left', bbox_to_anchor=(0.88, 1))
            plt.tight_layout()
            plt.show()
    
    # Create interactive widget
    interact_widget = widgets.interactive(
        update_plot,
        model=model_dropdown,
        self_corr=self_corr_dropdown,
        medium_hard=med_hard_dropdown,
        diff_plot=diff_plot_dropdown
    )
    
    # Display the widgets and output
    display(widgets.VBox([interact_widget, output_widget]))
    
    # Initial plot
    if models:
        update_plot(models[0], False, False, 'all')
