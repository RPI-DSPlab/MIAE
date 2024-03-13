data_aug=1 # 1 for data augmentation, 0 for no data augmentation

if [ "$data_aug" -eq 1 ]; then
    data_dir="/data/public/miae_experiment_aug_overfit/target"
else
    data_dir="/data/public/miae_experiment_overfit/target"
fi
mkdir -p "$data_dir"

if [ "$data_aug" -eq 1 ]; then
    eh_dir="/data/public/example_hardness_aug_overfit"
else
    eh_dir="/data/public/example_hardness_overfit"
fi
mkdir -p "$eh_dir"


#datasets=("cifar10" "cifar100" "cinic10")
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
#example_hardness=("il" "pd" "cs")
example_hardness=("il")

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

    mkdir -p "$eh_dir/$dataset"
    for arch in "${archs[@]}"; do
        mkdir -p "$eh_dir/$dataset/$arch"
        for eh in "${example_hardness[@]}"; do
            result_dir="$eh_dir/$dataset/$arch/$eh"
            # if the predictions are already saved, skip
            if [ -f "$result_dir/${eh}_score.pkl" ]; then
                echo "Scores already saved for $dataset $arch $eh"
                continue
            fi
            # if the preparation directory is not empty, delete it
            if [ -d "./$eh" ]; then
                rm -r "./$eh"
            fi

            mkdir -p "$result_dir"
            prepare_dir="./$eh"
            echo "Running $dataset $arch $eh"

            python obtain_example_hardness.py --dataset "$dataset" --model "$arch" --example_hardness "$eh" \
            --result_path "$result_dir" --preparation_path "$prepare_dir" \
            --dataset_path "$data_dir"  --epoch "$num_epoch"


            rm -r "./$eh"
        done
    done
done
