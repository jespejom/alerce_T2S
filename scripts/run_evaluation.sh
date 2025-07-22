#!/bin/bash
# Script to run ALeRCE Text-to-SQL evaluation
# Usage: ./run_evaluation.sh [config_file]

# Default config file
CONFIG_FILE="evaluation_config.json"

# If a config file is provided as argument, use that instead
if [ "$#" -ge 1 ]; then
    CONFIG_FILE=$1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found."
    exit 1
fi

# Set Python path to include project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Parse config file
DATA_PATH=$(jq -r '.data_path' "$CONFIG_FILE")
DATA_FORMAT=$(jq -r '.data_format' "$CONFIG_FILE")
MODEL_NAME=$(jq -r '.model_name' "$CONFIG_FILE")
SAVE_PATH=$(jq -r '.save_path' "$CONFIG_FILE")
EXP_NAME=$(jq -r '.exp_name' "$CONFIG_FILE")
PARALLEL=$(jq -r '.parallel // "false"' "$CONFIG_FILE")
NUM_PROCESSES=$(jq -r '.num_processes // 4' "$CONFIG_FILE")
ACCESS_TIME=$(jq -r '.access_time // 2' "$CONFIG_FILE")
N_TRIES=$(jq -r '.n_tries // 5' "$CONFIG_FILE")
ALIAS_HANDLING=$(jq -r '.alias_handling // "false"' "$CONFIG_FILE")
SELF_CORRECTION=$(jq -r '.self_correction' "$CONFIG_FILE")
# Create parallel flag if needed
PARALLEL_FLAG=""
if [ "$PARALLEL" = "true" ]; then
    PARALLEL_FLAG="--parallel"
fi

# Create alias handling flag if needed
ALIAS_FLAG=""
if [ "$ALIAS_HANDLING" = "true" ]; then
    ALIAS_FLAG="--alias_handling"
fi

echo "========================================="
echo "Starting evaluation for experiment: $EXP_NAME"
echo "Model: $MODEL_NAME"
echo "Data path: $DATA_PATH"
echo "Parallel processing: $PARALLEL with $NUM_PROCESSES processes"
echo "========================================="

# Run the evaluation
python llm/evaluation.py \
    --data_path "$DATA_PATH" \
    --data_format "$DATA_FORMAT" \
    --model_name "$MODEL_NAME" \
    --exp_name "$EXP_NAME" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --access_time "$ACCESS_TIME" \
    --n_tries "$N_TRIES" \
    $PARALLEL_FLAG \
    $ALIAS_FLAG \
    --self_correction "$SELF_CORRECTION" \

echo "========================================="
echo "Evaluation completed for experiment: $EXP_NAME"
echo "Results saved to: $SAVE_PATH/$MODEL_NAME/$EXP_NAME/eval_$EXP_NAME.json"
echo "========================================="
