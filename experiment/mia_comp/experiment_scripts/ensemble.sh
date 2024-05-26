#!/bin/bash

# This script is used for experiments regarding to inter-attack ensemble, there's these arguments:
# -train_shadow: Train shadow-target data
# -train_ensemble: Train ensemble methods on shadow-target data
# -run_ensemble: Run ensemble inference on target data
# -evaluation: Evaluate ensemble predictions

# Check if at least 1 argument is passed
if [ "$#" -lt 1 ]; then
    echo "Error: You must provide at least one argument."
    exit 1
fi

if $1 != "train_shadow" && $1 != "train_ensemble" && $1 != "run_ensemble" && $1 != "evaluation"; then
    echo "Error: Invalid argument."
    exit 1
fi

mias=("losstraj" "shokri" "yeom" "aug" "lira")
arch="resnet56"
dataset="cifar10"
seeds=(0 1 2 3 4 5)
seedlist=""
for seed in "${seeds[@]}"; do
    seedlist+="${seed} "
done
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
done
num_epoch=100
ensemble_method=("avg" "stacking")

preds_path="/data/public/comp_mia_data/miae_experiment_aug_more_target_data"

data_dir="${preds_path}/target/${dataset}"

shadow_target_dir="${preds_path}/shadow_target_model_and_data"

if [ ! -d "$shadow_target_dir" ]; then
    mkdir -p "$shadow_target_dir"
fi

mode=$1
if [ "$mode" == "train_shadow" ]; then
    echo "ENSEMBLE: Training shadow models"
    python3 ensemble.py \
    --mode "train_shadow" \
    --dataset "$dataset" \
    --target_model $arch \
    --target_epochs "$num_epoch" \
    --aux_set_path "$data_dir" \
    --shadow_save_path "$shadow_target_dir" \
    --target_model_path "$shadow_target_dir" \
    --device "cuda:1" \
    --seed 0
elif [ "$mode" == "train_ensemble" ]; then
    for method in "${ensemble_method[@]}"; do
      echo "ENSEMBLE: Training ensemble model"
      python3 ensemble.py \
      --mode "train_ensemble" \
      --target_model $arch \
      --dataset "$dataset" \
      --preds_path $preds_path \
      --ensemble_seeds $seedlist \
      --attacks $mialist \
      --ensemble_save_path "${preds_path}/ensemble_file_save/${dataset}" \
      --shadow_target_data_path "$shadow_target_dir" \
      --ensemble_method $method
    done

elif [ "$mode" == "run_ensemble" ]; then
    echo "ENSEMBLE: Running ensemble model"
    for method in "${ensemble_method[@]}"; do
        python3 ensemble.py \
        --mode "run_ensemble" \
        --target_model $arch \
        --dataset "$dataset" \
        --preds_path $preds_path \
        --ensemble_seeds $seedlist \
        --attacks $mialist \
        --ensemble_save_path "${preds_path}/ensemble_file_save" \
        --shadow_target_data_path "$shadow_target_dir" \
        --ensemble_method $method \
        --target_data_path ${preds_path}/target/${dataset} \
        --ensemble_result_path "${preds_path}/ensemble_result"
    done

elif [ "$mode" == "evaluation" ]; then
    echo "ENSEMBLE: Evaluating ensemble predictions"
      python3 ensemble.py \
      --mode "run_ensemble" \
      --target_model $arch \
      --dataset "$dataset" \
      --preds_path $preds_path \
      --ensemble_seeds $seedlist \
      --attacks $mialist \
      --ensemble_save_path "${preds_path}/ensemble_file_save" \
      --shadow_target_data_path "$shadow_target_dir" \
      --ensemble_method $method \
      --target_data_path ${preds_path}/target/${dataset} \
      --ensemble_result_path "${preds_path}/ensemble_result"
fi