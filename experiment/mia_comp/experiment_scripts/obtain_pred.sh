# This script is used to obtain the predictions of the attack on the target models
seed=0

if [ $# -eq 1 ]; then  # if the number of arguments is 1, the argument is the seed
    seed=$1
fi

echo "obtain_pred.sh seed = $seed"

data_dir="/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_3/target"

preds_dir="/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_3/preds_sd${seed}"
mkdir -p "$preds_dir"


#datasets=("cifar10" "cifar100" "cinic10")
datasets=("cifar10")
#archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
archs=("resnet56")
mias=("losstraj" "shokri" "yeom" "aug" "lira")

prepare_path="/data/public/prepare_sd${seed}"

target_model_path="$data_dir/target_models"

cd /home/wangz56/MIAE_training_dir/MIAE/experiment/mia_comp

conda activate conda-zhiqi

for dataset in "${datasets[@]}"; do
  # if assign different num_epoch for different dataset
  if [ "$dataset" == "cifar10" ]; then
    num_epoch=100
  elif [ "$dataset" == "cifar100" ]; then
    num_epoch=150
  elif [ "$dataset" == "cinic10" ]; then
    num_epoch=150
  fi

    for arch in "${archs[@]}"; do
      # for a given dataset and architecture, save the predictions
      mkdir -p "$preds_dir/$dataset/$arch"
        for mia in "${mias[@]}"; do
            result_dir="$preds_dir/$dataset/$arch/$mia"
            # if the predictions are already saved, skip
            if [ -f "$result_dir/pred_$mia.npy" ]; then
                echo "Predictions already saved for $dataset $arch $mia"
                continue
            fi
            # if the preparation directory is not empty, delete it
            if [ -d "$prepare_path" ]; then
                rm -r "$prepare_path"
            fi

            mkdir -p "$result_dir"
            prepare_dir="$prepare_path"
            echo "Running $dataset $arch $mia"
            target_model_save_path="$target_model_path/$dataset/$arch"

            python3 obtain_pred.py \
            --dataset "$dataset"\
            --target_model "$arch"\
            --attack "$mia"\
            --result_path "$result_dir"\
            --seed "$seed"\
            --delete-files "True" \
            --preparation_path "$prepare_dir" \
            --data_aug "False"  \
            --target_model_path "$target_model_save_path" \
            --attack_epochs "$num_epoch" \
            --target_epochs "$num_epoch" \
            --data_path "$data_dir" \
            --device "cuda:1"

            rm -r "$prepare_path"
        done
    done
done
