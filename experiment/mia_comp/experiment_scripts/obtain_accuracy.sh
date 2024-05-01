# This script generates the accuracy for the MIAE experiment

# Get the datasets, architectures, MIAs and categories
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri" "yeom")
processopt=("union" "intersection")
seeds=(0 1 2 3)
fprs=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)

# Prepare the parameter lists for the experiment
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
done

seedlist=""
for seed in "${seeds[@]}"; do
    seedlist+="${seed} "
done

fprlist=""
for fpr in "${fprs[@]}"; do
    fprlist+="${fpr} "
done

optlist=""
for opt in "${processopt[@]}"; do
    optlist+="${opt} "
done

experiment_dir="/home/public/comp_mia_data/miae_experiment_aug_more_target_data"
accuracy_dir="$experiment_dir/accuracy"
mkdir -p "$accuracy_dir"


# Check if directory creation was successful
if [ -d "$accuracy_dir" ]; then
    echo "Successfully created directory '$accuracy_dir'."
else
    echo "Error: Failed to create directory '$accuracy_dir'."
    exit 1
fi

cd /home/zhangc26/MIAE/experiment/mia_comp

conda activate rpidsp

# Run the experiment
for dataset in "${datasets[@]}"; do
    file_name="$accuracy_dir/accuracy_${dataset}.txt"
    if [ -f "$file_name" ]; then
        rm "$file_name"
    fi
    touch "$file_name"
    for arch in "${archs[@]}"; do
        python obtain_accuracy.py --dataset "$dataset" \
                                  --arch "$arch" \
                                  --attacks ${mialist} \
                                  --fpr ${fprlist} \
                                  --process_opt ${optlist} \
                                  --accuracy_path "$accuracy_dir/accuracy_${dataset}.txt" \
                                  --data_path "$experiment_dir" \
                                  --seeds ${seedlist}
    done
done