#!/bin/bash
# Script to run ALeRCE Text-to-SQL self-correction
# Usage: ./run_selfcorrection.sh [config_file]

# Default config file
CONFIG_FILE="selfcorrection_config.json"

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
MODEL_NAME=$(jq -r '.model_name' "$CONFIG_FILE")
DATA_PATH=$(jq -r '.data_path' "$CONFIG_FILE")
DATA_FORMAT=$(jq -r '.data_format' "$CONFIG_FILE")
EXP_NAME=$(jq -r '.exp_name' "$CONFIG_FILE")
SAVE_PATH=$(jq -r '.save_path' "$CONFIG_FILE")
EVAL=$(jq -r '.eval // "false"' "$CONFIG_FILE")
BATCH=$(jq -r '.batch // "false"' "$CONFIG_FILE")
PROMPT_SC=$(jq -r '.promptV_sc' "$CONFIG_FILE")
PARALLEL=$(jq -r '.parallel // "false"' "$CONFIG_FILE")
NUM_PROCESSES=$(jq -r '.num_processes // 4' "$CONFIG_FILE")
ACCESS_TIME=$(jq -r '.access_time // 2' "$CONFIG_FILE")
N_TRIES=$(jq -r '.n_tries // 5' "$CONFIG_FILE")
ALIAS_HANDLING=$(jq -r '.alias_handling // "false"' "$CONFIG_FILE")

# Create flags based on boolean values
EVAL_FLAG=""
if [ "$EVAL" = "true" ]; then
    EVAL_FLAG="--eval"
fi
BATCH_FLAG=""
if [ "$BATCH" = "true" ]; then
    BATCH_FLAG="--batch"
fi
PARALLEL_FLAG=""
if [ "$PARALLEL" = "true" ]; then
    PARALLEL_FLAG="--parallel"
fi
ALIAS_FLAG=""
if [ "$ALIAS_HANDLING" = "true" ]; then
    ALIAS_FLAG="--alias_handling"
fi

echo "========================================="
echo "Starting self-correction for experiment: $EXP_NAME"
echo "Model: $MODEL_NAME"
echo "Data path: $DATA_PATH"
echo "Evaluation after correction: $EVAL"
echo "Batch processing: $BATCH"
echo "========================================="

# Run the self-correction process
python llm/main_selfcorrection.py \
    --data_path "$DATA_PATH" \
    --data_format "$DATA_FORMAT" \
    --model_name "$MODEL_NAME" \
    --exp_name "$EXP_NAME" \
    --save_path "$SAVE_PATH" \
    --num_processes "$NUM_PROCESSES" \
    --access_time "$ACCESS_TIME" \
    --n_tries "$N_TRIES" \
    --promptV_sc "$PROMPT_SC" \
    $EVAL_FLAG \
    $BATCH_FLAG \
    $PARALLEL_FLAG \
    $ALIAS_FLAG

# Check if the self-correction was successful
if [ $? -ne 0 ]; then
    echo "Error: Self-correction process failed!"
    exit 1
fi

echo "========================================="
echo "Self-correction completed for experiment: $EXP_NAME"
echo "Results saved to: $SAVE_PATH/$MODEL_NAME/$EXP_NAME/corrected_$EXP_NAME.json"
if [ "$EVAL" = "true" ]; then
    echo "Evaluation results saved to: $SAVE_PATH/$MODEL_NAME/$EXP_NAME/eval_corrected_$EXP_NAME.json"
fi
echo "========================================="