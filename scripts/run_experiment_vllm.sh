#!/bin/bash
# Script to run ALeRCE Text-to-SQL experiment
# Usage:
#  vllm serve model/model_name
#  ./run_experiment_vllm.sh [options]

# Set Python path to include project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Default values
DATA_PATH="llm/data/txt2sql_alerce_train_v4_0.csv"
DATA_FORMAT="csv"
MODEL_NAME="Qwen/Qwen2.5-Coder-3B"
EXP_NAME="alerce_train_direct_v8"
SQL_GEN="direct"
N_EXPS=3
SAVE_PATH="./results"
EVAL="false"
SELF_CORRECTION="false"
SL_METHOD="true"
PROMPT_SL="sl_v3"
PROMPT_DC="diff_v8"
PROMPT_GEN="dir_v8"
PROMPT_SC="sc_v3"
NUM_PROCESSES=2
PARALLEL="true"
ACCESS_TIME=2
N_TRIES=7
ALIAS_HANDLING="true"
BATCH="false"
MODEL_NAME_DECOMP="null"
MAX_NEW_TOKENS=4000
MAX_NEW_TOKENS_DECOMP=4000
T=0.01

# Override defaults with command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --data_format)
            DATA_FORMAT="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --sql_gen_method)
            SQL_GEN="$2"
            shift 2
            ;;
        --n_exps)
            N_EXPS="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --eval)
            EVAL="true"
            shift
            ;;
        --self_correction)
            SELF_CORRECTION="true"
            shift
            ;;
        --schema_linking_method)
            SL_METHOD="$2"
            shift 2
            ;;
        --promptV_sl)
            PROMPT_SL="$2"
            shift 2
            ;;
        --promptV_diff_class)
            PROMPT_DC="$2"
            shift 2
            ;;
        --promptV_gen)
            PROMPT_GEN="$2"
            shift 2
            ;;
        --promptV_sc)
            PROMPT_SC="$2"
            shift 2
            ;;
        --num_processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="true"
            shift
            ;;
        --access_time)
            ACCESS_TIME="$2"
            shift 2
            ;;
        --n_tries)
            N_TRIES="$2"
            shift 2
            ;;
        --alias_handling)
            ALIAS_HANDLING="$2"
            shift 2
            ;;
        --batch)
            BATCH="true"
            shift
            ;;
        --model_name_decomp)
            MODEL_NAME_DECOMP="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --max_new_tokens_decomp)
            MAX_NEW_TOKENS_DECOMP="$2"
            shift 2
            ;;
        --t)
            T="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
    --t "$T"

# Check if the experiment was successful
if [ $? -ne 0 ]; then
    echo "Error: Experiment failed!"
    exit 1
fi

echo "========================================="
echo "Experiment completed: $EXP_NAME"
echo "Results saved to: $SAVE_PATH/$MODEL_NAME/$EXP_NAME"
echo "========================================="

