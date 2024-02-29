data_dir="/data/public/miae_experiment_1"
mkdir -p "$data_dir"

preds_dir="$data_dir/preds"
mkdir -p "$preds_dir"


#datasets=("cifar10" "cifar100" "cinic10")
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri")
seed=0

target_model_path="$data_dir/target_models"

cd /home/wangz56/MIAE/experiment/mia_comp

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

  mkdir -p "$preds_dir/$dataset"
  # save the dataset
  echo "Saving dataset $dataset"
  python3 obtain_pred.py --dataset "$dataset" --save_dataset "True" --result_path "$data_dir" --seed "$seed"
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
            if [ -d "./$mia" ]; then
                rm -r "./$mia"
            fi

            mkdir -p "$result_dir"
            prepare_dir="./$mia"
            echo "Running $dataset $arch $mia"

            python3 obtain_pred.py --dataset "$dataset" --target_model "$arch" --attack "$mia" \
            --result_path "$result_dir" --seed "$seed" --delete-files "True" --preparation_path "$prepare_dir" \
            --data_aug "False"  --target_model_path "$target_model_path" --attack_epochs "$num_epoch" \
            --target_epochs "$num_epoch"

            rm -r "./$mia"
        done
    done
done
