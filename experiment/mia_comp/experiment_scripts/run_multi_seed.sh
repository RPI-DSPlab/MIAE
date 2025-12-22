#!/bin/bash

# run it by: `bash run_multi_seed.sh {0..5}`
# List of arguments
seeds=("$@")

DATA_DIR="data"

script_out_dir=$DATA_DIR``

# for each seed
for sd in "${seeds[@]}"; do
    log_file="${script_out_dir}/output_${sd}.log"
    
    # Remove the log file if it exists
    if [ -f "$log_file" ]; then
        rm "$log_file"
        echo "Removed existing log file: $log_file"
    fi

    # Launch the experiment and save output to log file
    CUDA_VISIBLE_DEVICES=1 ./experiment_scripts/obtain_pred.sh "$sd" > "$log_file" 2>&1 &
done

# Wait for all background processes to complete
wait

echo "All tasks completed. Check output files in $script_out_dir"