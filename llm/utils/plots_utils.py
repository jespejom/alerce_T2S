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
        medium_hard: bool = False
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
                
            aggregate_metrics = metrics_aggregation(results=results)
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
    # change advanced to hard
    difficulties = [d.replace('advanced', 'hard') for d in difficulties]
    axs[1].set_xticklabels(difficulties)
    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0].legend(bbox_to_anchor=(1.02, 1.0))
    # set suptitle
    plt.suptitle('Query Generation Strategies', fontsize=24, y=0.95)
    # Add values on top of the bars
    for j, ax in enumerate(axs):
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # print(height)
            # Add the value on top of the bar, just above the standard bar height
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01 + std[j][i], f"{height:.3f}", ha='center', va='bottom')            
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.04)
    axs[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])    
    # Set title by each column
    plt.subplots_adjust(hspace=0.0, )
    # plt.tight_layout()
    # plt.xlabel('Difficulties')
    plt.legend( bbox_to_anchor=(1.02, 1.0),)
    plt.show()


def plot_perfect_match_by_difficulty_models(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_names: List[str],
        exp_names: List[str],
        std_dev: Union[str, None] = None,
        self_corr: bool = True,
        formatted_columns: bool = True,
        take_error_golds: bool = False,
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
    # change advanced to hard
    difficulties = [d.replace('advanced', 'hard') for d in difficulties]
    axs[1].set_xticklabels(difficulties)
    axs[0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0].legend(bbox_to_anchor=(1.02, 1.02))
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
    
    # Print tables in LaTeX format with enhanced formatting
    print("\n" + "="*80)
    print("PRECISION - ROWS - LaTeX Format")
    print("="*80)
    latex_str_precision_rows = create_enhanced_latex_table(
        precision_rows_df_formatted, 
        'Precision Scores for Rows',
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
        'Precision Scores for Columns', 
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
        'Recall Scores for Rows',
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
        'Recall Scores for Columns', 
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
        'F1-Scores for Rows',
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
        'F1-Scores for Columns', 
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
        formatted_columns: bool = True
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
        for j, (exp_label, exp_data) in enumerate(data.items()):
            oid_rates = [exp_data['oid'][d] for d in difficulties]
            column_rates = [exp_data['column'][d] for d in difficulties]
            # Plot OID matches
            axs[0, i].bar(x + (j - 0.5) * width, oid_rates, width, label=exp_label)
            # If std_dev is used, add error bars
            # Plot Column matches
            axs[1, i].bar(x + (j - 0.5) * width, column_rates, width, label=exp_label, )
            if std_dev:
                oid_std = [exp_data['oid_std'][d] for d in difficulties]
                # stds_.extend(oid_std)
                axs[0, i].errorbar(x + (j - 0.5) * width, oid_rates, yerr=oid_std, fmt='none', ecolor='black', capsize=5)
                column_std = [exp_data['column_std'][d] for d in difficulties]
                # stds_.extend(column_std)
                axs[1, i].errorbar(x + (j - 0.5) * width, column_rates, yerr=column_std, fmt='none', ecolor='black', capsize=5)
                for n in range(len(column_std)):
                    stds_.append(oid_std[n])
                    stds_.append(column_std[n])
            stds__.append(stds_)
        # Store the standard deviations for later use
        stds.extend(stds__)  # Mean of the standard deviations for each subplot
        # stds.append(stds__)  # Mean of the standard deviations for each subplot

    # Set x-ticks and labels
    axs[0, 0].set_ylabel('Rows', fontsize=20)
    axs[1, 0].set_ylabel('Columns', fontsize=20)
    
    axs[1, 0].set_xticks(x)
    # change advanced to hard
    difficulties = [d.replace('advanced', 'hard') for d in difficulties]
    axs[1, 0].set_xticklabels(difficulties)
    axs[0, 0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1, 0].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[0, 1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    axs[1, 1].set_ylim(0, 1)  # Set y-axis limit to [0, 1] for percentage
    # Remove numbers from the y-axis of the second column
    axs[0, 1].yaxis.set_visible(False)
    axs[1, 1].yaxis.set_visible(False)
    # remove the 0 value from the y-axis of the first plot
    axs[0, 0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])    
    # Set title by each column
    axs[0, 0].set_title('Without Self-Correction', fontsize=18)
    axs[0, 1].set_title('With Self-Correction', fontsize=18)

    # add legend to the first column
    axs[0, 1].legend(bbox_to_anchor=(1.02, 1.0))
    # Remove space between the two rows of subplots
    plt.subplots_adjust(hspace=0.0, )

    # set suptitle
    plt.suptitle('Query Generation Strategies: Self-Correction', fontsize=24, y=0.95)
    # Add values on top of the bars
    for j, ax in enumerate(axs.flatten()):
        for i, bar in enumerate(ax.patches):
            height = bar.get_height()
            # Add the value on top of the bar
            # add the value on top of the bar, just above the standard deviation bar height
            ax.text(bar.get_x() + bar.get_width()/2, height +0.01+ stds[j][i], f"{height:.2f}", ha='center', va='bottom')
    # add general y label
    fig.supylabel('% Perfect Matching Queries', fontsize=24, x=0.02)
    # plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1.0),)
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
                axs[i].annotate(f'{height:.1f}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points',
                                fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
    axs[0].set_xlabel("Without Self-Correction", fontsize=18)
    if percentage: 
        axs[0].set_ylabel("Percentage from Total Queries (%)", fontsize=18)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
    else: axs[0].set_ylabel("Error Count", fontsize=18)
    axs[1].set_xlabel("With Self-Correction", fontsize=18)
    axs[0].yaxis.set_visible(True)
    axs[1].yaxis.set_visible(True)
    plt.subplots_adjust(hspace=0.02, )
    plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    plt.suptitle(f"Execution Errors by Generation Method", fontsize=20)
    plt.ylabel("Error Count")
    plt.tight_layout()
    plt.show()

def plot_execution_errors_by_model_by_difficulty(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: str,
        self_corr: bool = True,
        percentage: bool = False,
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
    if self_corr: self_corr_key = 'self_corrected'
    else: self_corr_key = 'corrected'
    
    experiment_labels = {}
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
            for res in results:
                error_info = res['comparison'].get('error_pred', None)
                difficulty = res.get('difficulty')
                if difficulty not in experiment_labels[exp_label]:
                    experiment_labels[exp_label][difficulty] = []
                if error_info:
                    error_class = get_error_class(error_info).get('error_class')
                    experiment_labels[exp_label][difficulty].append(error_class)
                else:
                    experiment_labels[exp_label][difficulty].append(0)  # No error
                    continue
        
        except KeyError as e:
            print(f"Warning: Missing data for {model_name} - {experiment}: {e}")
    print(experiment_labels)
    difficulties = sorted(set(
        difficulty for data in experiment_labels.values() for difficulty in data.keys()
    ), key=lambda x: ['simple', 'medium', 'advanced'].index(x) if x in ['simple', 'medium', 'advanced'] else 3)
    # change advanced to hard
    # difficulties = [d.replace('advanced', 'hard') for d in difficulties]
    fig, axs = plt.subplots(1, len(difficulties), figsize=(20, 10), sharex=True, sharey=True)
    for i, difficulty in enumerate(difficulties):
        error_data = []
        for exp_label, data in experiment_labels.items():
            if difficulty in data:
                error_counts = pd.Series(data[difficulty]).value_counts().reset_index()
                # Change "QueryCanceled" to "Timeout"
                error_counts['index'] = error_counts['index'].replace({
                    'QueryCanceled': 'Timeout',})
                error_counts.columns = ['error_type', 'error_count']
                error_counts['experiment'] = exp_label
                if len(error_data) == 0:
                    error_data = error_counts
                else:
                    error_data = pd.concat([error_data, error_counts], ignore_index=True)
        # Plot the errors for each difficulty
        if error_data.empty:
            print(f"No errors found for difficulty {difficulty} in model {model_name}.")
            continue
        if percentage:
            total_errors = error_data['error_count'].sum()
            error_data['error_count'] = (error_data['error_count'] / total_errors) * 100
        # remove rows with error_type 0
        error_data = error_data[error_data['error_type'] != 0]

        # Group the error types by their class and by experiment, to plot them without stacking
        error_data = error_data.groupby(['error_type', 'experiment']).sum().reset_index()
        sns.barplot(x='error_type', y='error_count', data=error_data, ax=axs[i], hue='experiment')
        if percentage:
            # add value labels on top of each bar
            for p in axs[i].patches:
                height = p.get_height()
                axs[i].annotate(f'{height:.1f}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points',
                                fontsize=12)
        axs[i].set_title(f"{difficulty.replace('advanced','hard')}", fontsize=16)
        axs[i].set_xlabel("Error Type", fontsize=14)
        if percentage: 
            axs[i].set_ylabel("Percentage from Total Queries (%)", fontsize=14)
            axs[i].set_ylim(0, 100)
        else: axs[i].set_ylabel("Error Count", fontsize=14)
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right', fontsize=14)
        # axs[i].set_ylim(0, error_data['error_count'].max() * 1.1)  # Set y-axis limit to 10% above max count
        axs[i].legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
    # axs[0].yaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    axs[2].yaxis.set_visible(False)
    plt.subplots_adjust(hspace=0.01, )
    # plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
    # axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
    plt.suptitle(f"Execution Errors by difficulty", fontsize=20)
    plt.ylabel("Error Count")
    plt.tight_layout()
    plt.show()

def plot_execution_errors_by_model_type(
        evaluation_results: Dict[str, Dict[str, Any]],
        model_name: str,
        exp_names: str,
        percentage: bool = False,
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
                height = p.get_height()
                axs[i].annotate(f'{height:.1f}',
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points',
                                fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.legend(bbox_to_anchor=(1.02, 1.0), fontsize=12)
    axs[0].set_xlabel("Without Self-Correction", fontsize=18)
    if percentage: 
        axs[0].set_ylabel("Percentage from Total Queries (%)", fontsize=18)
        axs[0].set_ylim(0, 100)
        axs[1].set_ylim(0, 100)
    else: axs[0].set_ylabel("Error Count", fontsize=18)
    axs[1].set_xlabel("With Self-Correction", fontsize=18)
    axs[1].yaxis.set_visible(True)
    plt.subplots_adjust(hspace=0.02, )
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    plt.suptitle(f"Execution Errors Type by Generation Method", fontsize=20)
    plt.ylabel("Error Count")
    plt.tight_layout()
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
