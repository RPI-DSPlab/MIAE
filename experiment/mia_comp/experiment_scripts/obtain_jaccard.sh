# Description:
# This script is used to obtain the Jaccard similarity between the TPR of the MIAE and standard attack models for different FPRs.

# modify this to set up directory:
DATA_DIR="data"

# Configuration
datasets=("cifar100" "cifar10")
archs=("resnet56")
seeds_for_file=(0 1 2 3)
fprs=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)
experiment_dir="${DATA_DIR}/repeat_miae_standard_exp"
plot_dir="$experiment_dir/jaccard_similarity"
mkdir -p "$plot_dir"

# Main loop: iterate over all seeds, datasets, and architectures to obtain the Jaccard similarity
for seed in "${seeds_for_file[@]}"; do
    for dataset in "${datasets[@]}"; do
        for arch in "${archs[@]}"; do
            base_dirs=()
            dir_path="${experiment_dir}/miae_standard_exp_${seed}/graphs/venn/fpr/pairwise/${dataset}/${arch}/TPR"

            if [ -d "$dir_path" ]; then
                base_dirs+=("$dir_path")
            else
                echo "Warning: Directory $dir_path does not exist."
            fi

            base_dirs_string=$(printf " %s" "${base_dirs[@]}")
            base_dirs_string=${base_dirs_string:1}

            python ../obtain_jaccard.py --fpr "${fprs[@]}" \
                                        --base_dir "${base_dirs_string}" \
                                        --plot_dir "${plot_dir}/${dataset}/${arch}"
        done
    done
done