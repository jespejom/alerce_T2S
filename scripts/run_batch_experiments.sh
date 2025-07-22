#!/bin/bash
# Master script to run batch experiments with different configurations

# Create directories if they don't exist
mkdir -p logs

# List of config files to run
CONFIG_FILES=(
    "experiment_config_opus_direct.json"
    "experiment_config_opus_stepbystep.json"
    "experiment_config_sonnet_direct.json"
)

# Run each experiment in sequence
for config in "${CONFIG_FILES[@]}"; do
    echo "Starting experiment with config: $config"
    
    # Run experiment and save log
    ./run_experiment.sh "$config" 2>&1 | tee "logs/experiment_$(basename "$config" .json).log"
    
    # Check if experiment was successful
    if [ $? -eq 0 ]; then
        echo "Experiment $config completed successfully!"
        
        # Extract experiment name from config
        EXP_NAME=$(jq -r '.exp_name' "$config")
        
        # Create evaluation config based on experiment config
        EVAL_CONFIG="evaluation_${EXP_NAME}.json"
        
        # Copy relevant fields from experiment config to evaluation config
        jq '{
            data_path: .data_path,
            data_format: .data_format,
            model_name: .model_name,
            save_path: .save_path,
            exp_name: .exp_name,
            parallel: true,
            num_processes: 4,
            access_time: 2,
            n_tries: 5,
            alias_handling: true
        }' "$config" > "$EVAL_CONFIG"
        
        # Run evaluation
        ./run_evaluation.sh "$EVAL_CONFIG" 2>&1 | tee "logs/eval_${EXP_NAME}.log"
    else
        echo "Experiment $config failed!"
    fi
    
    echo "============================================"
done

echo "All experiments completed!"
