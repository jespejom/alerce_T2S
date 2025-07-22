#!/bin/bash
# Script to run ALeRCE Text-to-SQL experiment
# Usage: ./run_experiment.sh [config_file]

# Default config file
CONFIG_FILE="experiment_config.json"

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
EXP_NAME=$(jq -r '.exp_name' "$CONFIG_FILE")
SQL_GEN=$(jq -r '.sql_gen_method' "$CONFIG_FILE")
N_EXPS=$(jq -r '.n_exps' "$CONFIG_FILE")
SAVE_PATH=$(jq -r '.save_path' "$CONFIG_FILE")
EVAL=$(jq -r '.eval // "false"' "$CONFIG_FILE")
SELF_CORRECTION=$(jq -r '.self_correction // "false"' "$CONFIG_FILE")
SL_METHOD=$(jq -r '.schema_linking_method // "true"' "$CONFIG_FILE")
PROMPT_SL=$(jq -r '.promptV_sl' "$CONFIG_FILE")
PROMPT_DC=$(jq -r '.promptV_diff_class' "$CONFIG_FILE")
PROMPT_GEN=$(jq -r '.promptV_gen' "$CONFIG_FILE")
PROMPT_SC=$(jq -r '.promptV_sc' "$CONFIG_FILE")
# Evaluation parameters
NUM_PROCESSES=$(jq -r '.num_processes // 8' "$CONFIG_FILE")
PARALLEL=$(jq -r '.parallel // "false"' "$CONFIG_FILE")
ACCESS_TIME=$(jq -r '.access_time // 2' "$CONFIG_FILE")
N_TRIES=$(jq -r '.n_tries // 5' "$CONFIG_FILE")
ALIAS_HANDLING=$(jq -r '.alias_handling // "true"' "$CONFIG_FILE")
# Default values for optional parameters
BATCH=$(jq -r '.batch // "false"' "$CONFIG_FILE")
MODEL_NAME_DECOMP=$(jq -r 'if has("model_name_decomp") then .model_name_decomp else "null" end' "$CONFIG_FILE")
# Model parameters
MAX_NEW_TOKENS=$(jq -r '.max_new_tokens // 4000' "$CONFIG_FILE")
MAX_NEW_TOKENS_DECOMP=$(jq -r '.max_new_tokens_decomp // 4000' "$CONFIG_FILE")
T=$(jq -r '.t // 0.0' "$CONFIG_FILE")

# Create evaluation flag if needed
EVAL_FLAG=""
if [ "$EVAL" = "true" ]; then
    EVAL_FLAG="--eval"
fi
# Create self-correction flag if needed
SELF_CORRECTION_FLAG=""
if [ "$SELF_CORRECTION" = "true" ]; then
    SELF_CORRECTION_FLAG="--self_correction"
fi
# Create batch flag string if needed
BATCH_FLAG=""
if [ "$BATCH" = "true" ]; then
    BATCH_FLAG="--batch"
fi

# Create decomp model string if needed
DECOMP_MODEL=""
if [ "$MODEL_NAME_DECOMP" != "null" ]; then
    DECOMP_MODEL="--model_name_decomp $MODEL_NAME_DECOMP"
fi

echo "========================================="
echo "Starting experiment: $EXP_NAME"
echo "Model: $MODEL_NAME"
echo "Data path: $DATA_PATH"
echo "========================================="

# Run the experiment
python llm/main.py \
    --data_path "$DATA_PATH" \
    --data_format "$DATA_FORMAT" \
    --model_name "$MODEL_NAME" \
    --exp_name "$EXP_NAME" \
    --sql_gen_method "$SQL_GEN" \
    --n_exps "$N_EXPS" \
    --save_path "$SAVE_PATH" \
    $EVAL_FLAG \
    $SELF_CORRECTION_FLAG \
    --schema_linking_method "$SL_METHOD" \
    --promptV_sl "$PROMPT_SL" \
    --promptV_diff_class "$PROMPT_DC" \
    --promptV_gen "$PROMPT_GEN" \
    --promptV_sc "$PROMPT_SC" \
    --num_processes "$NUM_PROCESSES" \
    --parallel "$PARALLEL" \
    --access_time "$ACCESS_TIME" \
    --n_tries "$N_TRIES" \
    --alias_handling "$ALIAS_HANDLING" \
    $BATCH_FLAG \
    $DECOMP_MODEL \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_new_tokens_decomp "$MAX_NEW_TOKENS_DECOMP" \
    --t "$T" \

# Check if the experiment was successful
if [ $? -ne 0 ]; then
    echo "Error: Experiment failed!"
    exit 1
fi

echo "========================================="
echo "Experiment completed: $EXP_NAME"
echo "Results saved to: $SAVE_PATH/$MODEL_NAME/$EXP_NAME"
echo "========================================="

