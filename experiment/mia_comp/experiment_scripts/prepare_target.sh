# This script is used to partition the dataset into target dataset and shadow dataset, then train the target model
seed=0
# for repeat training, we do shuffle_seed from 2 to 5
shuffle_seed=4
data_dir="/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_0/target"
mkdir -p "$data_dir"


datasets=("cifar10" "cifar100")
#datasets=("cifar10")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")

prepare_path="/data/public/prepare_sd${seed}"

target_model_path="$data_dir/target_models"

cd /home/wangz56/MIAE_training_dir/MIAE/experiment/mia_comp

conda activate conda-zhiqi

for dataset in "${datasets[@]}"; do
  # if assign different num_epoch for different dataset
  if [ "$dataset" == "cifar10" ]; then
    num_epoch=60
  elif [ "$dataset" == "cifar100" ]; then
    num_epoch=100
  elif [ "$dataset" == "cinic10" ]; then
    num_epoch=60
  fi

  mkdir -p "$data_dir/$dataset"
  # save the dataset
  echo "Saving dataset $dataset"
#  python3 obtain_pred.py --dataset "$dataset" --save_dataset "True" --data_path "$data_dir" --seed "$seed" --data_aug "True" --shuffle_seed "$shuffle_seed"
    for arch in "${archs[@]}"; do
      # for each arch, train the target model
      mkdir -p "$target_model_path/$dataset/$arch"
      target_model_save_path="$target_model_path/$dataset/$arch"
      echo "Obtaining target_model for $dataset $arch"
      # if the target model is already trained, then skip
      if [ -f "$target_model_save_path/target_model_$arch$dataset.pkl" ]; then
        echo "Target model for $dataset $arch already exists, skip"
        continue
      fi
      python3 obtain_pred.py --train_target_model "True" --dataset "$dataset" --target_model "$arch" \
       --seed "$seed" --delete-files "True" --data_aug "True"  --target_model_path "$target_model_save_path" \
       --attack_epochs "$num_epoch" --target_epochs "$num_epoch" --data_path "$data_dir" --shuffle_seed "$shuffle_seed"
    done
done
