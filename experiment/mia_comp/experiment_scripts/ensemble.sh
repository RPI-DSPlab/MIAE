#!/bin/bash

# Check if at least 1 argument is passed
if [ "$#" -lt 1 ]; then
    echo "Error: You must provide at least one argument."
    exit 1
fi

if $1 != "train_shadow" && $1 != "train_ensemble" && $1 != run_ensemble; then
    echo "Error: Invalid argument."
    exit 1
fi

mias=("losstraj" "shokri" "yeom" "aug")
archs=("resnet56")
dataset="cifar10"
num_epoch=100

data_dir="/data/public/comp_mia_data/miae_experiment_aug_more_target_data/target/${dataset}"

shadow_target_dir="/data/public/comp_mia_data/miae_experiment_aug_more_target_data/shadow_target_model_and_data"
if [ ! -d "$shadow_target_dir" ]; then
    mkdir -p "$shadow_target_dir"
fi

mode=$1
if [ "$mode" == "train_shadow" ]; then
    echo "ENSEMBLE: Training shadow models"
    python3 obtain_pred.py \
    --dataset "$dataset" \
    --target_model "resnet56" \
    --num_epoch "$num_epoch" \
    --target_epochs "$num_epoch" \
    --aux_set_path "$data_dir" \
    --shadow_save_path "$shadow_target_dir" \
    --device "cuda:1" \
    --seed 0
elif [ "$mode" == "train_ensemble" ]; then
    echo "ENSEMBLE: Training ensemble model"
    python3 train_ensemble.py
elif [ "$mode" == "run_ensemble" ]; then
    echo "ENSEMBLE: Running ensemble model"
    python3 run_ensemble.py
fi