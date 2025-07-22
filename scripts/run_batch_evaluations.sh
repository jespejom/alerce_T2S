#!/bin/bash
# Script to run multiple ALeRCE Text-to-SQL evaluations in sequence
# Usage: ./run_batch_evaluations.sh [batch_config_file]

# Default batch config file
BATCH_CONFIG_FILE="batch_evaluation_config.json"

# If a config file is provided as argument, use that instead
if [ "$#" -ge 1 ]; then
    BATCH_CONFIG_FILE=$1
fi

# Check if batch config file exists
if [ ! -f "$BATCH_CONFIG_FILE" ]; then
    echo "Error: Batch config file '$BATCH_CONFIG_FILE' not found."
    exit 1
fi

# Set Python path to include project root
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Get the number of experiments in the batch
NUM_EXPERIMENTS=$(jq '.experiments | length' "$BATCH_CONFIG_FILE")

echo "========================================="
echo "Starting batch evaluation with $NUM_EXPERIMENTS experiment(s)"
echo "========================================="

# Loop through each experiment in the batch
for i in $(seq 0 $(($NUM_EXPERIMENTS-1)))
do
    # Extract the experiment config to a temporary file
    jq -r ".experiments[$i]" "$BATCH_CONFIG_FILE" > temp_config.json
    
    # Get experiment name for logging
    EXP_NAME=$(jq -r '.exp_name' temp_config.json)
    MODEL_NAME=$(jq -r '.model_name' temp_config.json)
    
    echo "----------------------------------------"
    echo "Running experiment $((i+1)) of $NUM_EXPERIMENTS: $EXP_NAME with model $MODEL_NAME"
    echo "----------------------------------------"
    
    # Run the evaluation script with the temporary config
    ./scripts/run_evaluation.sh temp_config.json
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "Experiment $EXP_NAME completed successfully"
    else
        echo "Warning: Experiment $EXP_NAME may have encountered issues"
    fi
done

# Clean up temporary file
rm -f temp_config.json

echo "========================================="
echo "Batch evaluation completed for all $NUM_EXPERIMENTS experiment(s)"
echo "========================================="
