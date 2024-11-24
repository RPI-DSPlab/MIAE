# modify this to set up directory:
DATA_DIR="/home/data/wangz56"

datasets=("cifar10")
archs=("resnet56")
seeds_for_file=(0 1 2 3)
option=("TPR")
fprs=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)
experiment_dir="${DATA_DIR}/repeat_exp_set"
plot_dir="$experiment_dir/jaccard_similarity"
mkdir -p "$plot_dir"

for fpr in "${fprs[@]}"; do
    mkdir -p "${plot_dir}/fpr_${fpr}"
done

# Generate seed list
seedlist=""
for seed in "${seeds_for_file[@]}"; do
    seedlist+="${seed} "
done

# Initialize an empty list
base_dirs=()

# Construct the full paths
for seed in "${seeds_for_file[@]}"; do
    for dataset in "${datasets[@]}"; do
        for arch in "${archs[@]}"; do
            tmp_dir="${experiment_dir}/miae_experiment_aug_more_target_data_${seed}/graphs_rebuttal/instances3/venn/fpr/pairwise/${dataset}/${arch}/TPR"
            if [ -d "$tmp_dir" ]; then
                base_dirs+=("$tmp_dir")
            else
                echo "Directory $tmp_dir does not exist."
            fi
        done
    done
done

# Join the base directories into a single string separated by spaces
base_dirs_string=$(printf " %s" "${base_dirs[@]}")
base_dirs_string=${base_dirs_string:1}  # Remove leading space


python ../obtain_jaccard.py --fpr "${fprs[@]}" \
                         --base_dir "${base_dirs_string}" \
                         --plot_dir "${plot_dir}"
