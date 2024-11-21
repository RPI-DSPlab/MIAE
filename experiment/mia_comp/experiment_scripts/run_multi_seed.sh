#!/bin/bash

# run it by bash run_multi_seed.sh {0..5}
# List of arguments
seeds=("$@")

# for each seed
for sd in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES=0 ./experiment_scripts/obtain_pred.sh "$sd"
done
