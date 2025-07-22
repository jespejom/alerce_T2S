# ALeRCE Text-to-SQL (T2S) System

A system for converting natural language queries to SQL for astronomical data using Large Language Models.

## Overview

The ALeRCE T2S system provides a pipeline for converting natural language queries about astronomical data into executable SQL queries. The system leverages Large Language Models (LLMs) and implements various strategies to optimize query generation based on query complexity.

## Architecture

The system follows a modular architecture with the following components:

```
alerce_T2S/
├── llm/                    # Core LLM components for text-to-SQL conversion
│   ├── main.py            # Main entry point for running experiments
│   ├── classification/    # Query classification components
│   ├── schema/            # Schema linking components
│   ├── sql_gen/           # SQL generation components
│   ├── model/             # LLM model wrappers
│   └── prompts/           # Prompt templates
├── utils/                 # Utility functions and helpers
├── results/               # Experiment results
├── docs/                  # Documentation and reference materials
└── run/                   # Scripts to run the pipeline
```

## Core Components

### Schema Linking

The `SchemaLinking` class identifies which database tables are relevant to a given natural language query. This ensures that only necessary tables are provided to the model for SQL generation, improving both efficiency and accuracy.

### Query Classification

Two classification components analyze the query:

1. **Difficulty Classification** (`DifficultyClassification`): Categorizes queries as simple, medium, or advanced based on their complexity.
2. **Q3C Classification** (`Q3cClassification`): Identifies whether a query requires spatial indexing operations.

### SQL Generation

The system supports multiple SQL generation strategies:

1. **Direct Generation** (`DirectSQLGenerator`): Converts the natural language query directly to SQL in a single step.
2. **Step-by-Step Generation** (`StepByStepSQLGenerator`): For complex queries, first creates a logical plan then converts the plan to SQL.

### Model Interface

The `LLMModel` interface provides a standardized way to interact with different language models. Currently implemented:

- `GPTModel`: Interface for OpenAI GPT models
- `ClaudeModel`: Interface for Anthropic Claude models
- Additional models (Llama, Qwen) in development

## Workflow

1. **Schema Linking**: Identify relevant tables for the query
2. **Query Classification**: Determine query difficulty and whether it requires spatial operations
3. **SQL Generation**: Use the appropriate generation strategy based on query classification
4. **Evaluation**: Compare generated SQL against gold standard queries
5. **Self-Correction** (optional): Apply corrections to improve query accuracy

## SQL Query Evaluation System

The evaluation system for SQL queries is designed to provide comprehensive metrics on the quality of generated queries compared to gold standard queries. It focuses on two key aspects:

### Features

- **Multi-mode processing**: Support for both sequential and parallel evaluation
- **Identifier matching**: Smart matching of row identifiers (oid, catalog_id, etc.) with customizable alias handling
- **Column comparison**: Evaluation of selected columns between gold and predicted queries
- **Flexible input formats**: Support for DataFrames, dictionaries, JSON files, and CSV files
- **Markdown parsing**: Automatic extraction of SQL queries from markdown code blocks
- **Execution metrics**: Collection of execution times and success rates
- **Duplicate detection**: Optimization through identifying and reusing results for duplicate queries
- **Comprehensive error handling**: Detailed error reporting for failed queries

### Key Components

- **`compare_results_oids`**: Evaluates identifier matches between gold and predicted query results
- **`compare_results_columns`**: Compares column selections between gold and predicted queries
- **`compare_sql_queries`**: Compares a single predicted query against a gold query
- **`compare_sql_queries_list`**: Handles multiple predicted queries for a single gold query
- **`alerce_usercases_evaluation`**: Main evaluation function for processing datasets of queries
- **`evaluation.py`**: Unified interface with helper functions for both sequential and parallel evaluation

### Metrics

The system calculates the following metrics for both identifiers and columns:

- **Precision**: Proportion of predicted items that are correct
- **Recall**: Proportion of gold standard items that were correctly predicted
- **F1 Score**: Harmonic mean of precision and recall
- **Perfect Match**: Binary indicator of exact matches (for identifiers: both precision and recall = 1.0; for columns: recall = 1.0)
- **Perfect Match Rate**: Proportion of queries with perfect matches
- **True positives**: Items correctly identified in the prediction
- **False positives**: Items incorrectly included in the prediction
- **False negatives**: Gold standard items missing from the prediction

### Usage

```python
from evaluation import evaluate_alerce_queries

# Sequential evaluation
results = evaluate_alerce_queries(
    database="path/to/gold_queries.csv",
    predicted_sqls=predicted_queries_dict,
    parallel=False,
    access_time=2,
    n_tries=3
)

# Parallel evaluation
results = evaluate_alerce_queries(
    database="path/to/gold_queries.csv",
    predicted_sqls=predicted_queries_dict,
    parallel=True,
    num_processes=8,
    access_time=2,
    n_tries=3
)
```

## Usage

### Running Experiments

```bash
python llm/main.py --data_path /path/to/queries.csv \
                  --model_name gpt-4 \
                  --exp_name experiment1 \
                  --sql_gen step-by-step \
                  --save_path ./results \
                  --n_exps 5 \
                  --eval true \
                  --self_correction true
```

You can control the SQL generation method with `--sql_gen` parameter:
- `direct`: Uses direct SQL generation for all queries regardless of difficulty
- `step-by-step`: Uses difficulty classification, applying direct generation for simple queries and step-by-step for medium/advanced queries

The experiment results will be saved in: `./results/model_name/experiment_name/`
The folder will contain:
- `config.json`: Parameters, arguments, and prompt templates used in the experiment
- `experiments.json`: Complete results of all experimental runs
- `evaluation.json`: Query evaluation metrics (if `--eval true`)
- `corrected_experiment.json`: Self-correction results (if `--self_correction true`)
- `evaluation_sc.json`: Evaluation metrics for self-corrected queries

### Running SQL Evaluations

```bash
# Run example evaluation comparing sequential and parallel methods
python llm/example_evaluation.py

# Custom evaluation script
python -c "
from llm.eval.evaluation import evaluate_alerce_queries
import pandas as pd
import json

# Load test data
test_data = pd.read_csv('llm/data/txt2sql_alerce_test_v4_0.csv')
test_data = test_data.rename(columns={'query': 'gold_query'})

# Load predictions (example)
with open('path/to/predictions.json', 'r') as f:
    predictions = json.load(f)
    
# Run evaluation
results = evaluate_alerce_queries(
    database=test_data,
    predicted_sqls=predictions,
    parallel=True,
    num_processes=8,
    save_path='evaluation_results/latest_eval'
)

print(f'Overall F1 score: {results[\"aggregate_metrics\"][\"oids\"][\"f1_score\"]:.4f}')
print(f'Perfect match rate: {results[\"aggregate_metrics\"][\"oids\"][\"perfect_match_rate\"]:.4f}')
"
```

## Current Status

### ✅ Done

- Basic repository structure established
- Schema linking implementation
- Query classification (difficulty and spatial)
- Direct SQL generation implementation
- Step-by-Step SQL generation implementation
- Model interface for OpenAI GPT and Anthropic Claude models
- Self-correction implementation

- Improved error handling and input validation across all modules
- Security improvements (removed hardcoded credentials)
- Added comprehensive type hints throughout the codebase
- Fixed relative imports for better code organization
- Added parameter validation in core modules
- Enhanced API error handling with specialized exception categories
- Improved retry mechanisms for API calls with exponential backoff
- Database connection best practices (connection pooling, disposal)

- **Comprehensive SQL Query Evaluation System:**
  - Sequential and parallel evaluation of SQL queries
  - Advanced identifier/column matching with alias handling 
  - Detailed metrics (precision, recall, F1, perfect match) for result comparison
  - Perfect match metrics for exact identifier and column matching
  - Support for various input formats (DataFrame, CSV, JSON)
  - Robust error handling and duplicate detection
  - Performance optimization through parallel processing

### 🔄 In Progress

- Support for additional LLM models (Llama, Qwen)
- Comprehensive testing framework
- Documentation improvements
- Implementing consistent error handling and logging across all modules

### 📝 TODO

- Standardize naming conventions across modules
- Add config file for centralized configuration management
- Implement logging system for better debugging
- Create prompt versioning and management system
- Add test cases and validation mechanisms
- Create high-level documentation for end-to-end workflow
- Build an interactive demo/UI
- Add support for local LLMs
- Implement a caching system for model calls to reduce API usage
- Add prompt recovery mechanism for interrupted experiments

## Contributing

Please follow these guidelines when contributing to the project:

1. Use consistent naming conventions
2. Include docstrings for all classes and methods
3. Add type hints to function signatures
4. Write tests for new functionality
5. Update documentation as needed

## License

[Add appropriate license information here]