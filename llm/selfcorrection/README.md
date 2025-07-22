# ALeRCE Text-to-SQL Self-Correction Module

This module provides functionality for correcting SQL queries that failed during execution in the ALeRCE text-to-SQL system. The self-correction approach uses error messages from the database to guide the correction of the SQL queries, either sequentially or in batch mode.

## Features

- **Sequential Self-Correction**: Process and correct SQL queries one by one
- **Batch Self-Correction**: Process multiple SQL corrections in parallel for improved throughput
- **Automatic Fallback**: If batch processing fails, the system falls back to sequential processing
- **Evaluation Integration**: Automatically evaluate corrected queries and compare with original results
- **Error Analysis**: Generate detailed reports showing improvements after correction

## Usage

### Command Line

You can use the self-correction module directly from the command line:

```bash
python main_selfcorrection.py --data_path DATA_PATH --model_name MODEL_NAME --exp_name EXP_NAME [OPTIONS]
```

Or use the convenience script:

```bash
./run_selfcorrection.sh --data_path DATA_PATH --model_name MODEL_NAME --exp_name EXP_NAME [OPTIONS]
```

### Required Arguments

- `--data_path`: Path to the dataset file
- `--model_name`: Name or path of the model to use for self-correction
- `--exp_name`: Name of the original experiment to correct

### Optional Arguments

- `--data_format`: Format of the dataset (json, csv, etc.) [default: csv]
- `--save_path`: Base path where results are saved [default: ./results]
- `--no-eval`: Disable evaluation after correction
- `--batch`: Use batch mode for self-correction
- `--num_processes`: Number of processes for parallel evaluation
- `--no-parallel`: Disable parallel processing for evaluation
- `--access_time`: The access time for the database [default: 2]
- `--n_tries`: The number of tries to run each query [default: 5]
- `--no-alias_handling`: Disable column name alias handling

## Integration with Main Pipeline

The self-correction module is integrated with the main text-to-SQL pipeline in `main.py`. To use it:

1. Run a normal text-to-SQL experiment with evaluation:
```bash
python main.py --data_path DATA_PATH --model_name MODEL_NAME --exp_name EXP_NAME --eval
```

2. Then add the `--self_correction` flag to enable automatic correction of failed queries:
```bash
python main.py --data_path DATA_PATH --model_name MODEL_NAME --exp_name EXP_NAME --eval --self_correction
```

## Output Files

When running self-correction, the following files are generated:

- `corrected_EXP_NAME.json`: Contains the corrected SQL queries
- `eval_corrected_EXP_NAME.json`: Contains evaluation results for the corrected queries
- Updated `config.json`: Contains settings used for self-correction

## Example

```bash
# Run self-correction on an existing experiment
./run_selfcorrection.sh --data_path ./data/txt2sql_alerce_test_v4_0.csv --model_name gpt-4.1-nano \
  --exp_name alerce_direct_test_dummy --batch
```
