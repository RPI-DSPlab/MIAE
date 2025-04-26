# Description: This script is used to generate the upset plots for the MIAE experiment.

# modify this to set up directory:
DATA_DIR="data"
datasets=("cifar10")
archs=("resnet56")
mias=("losstraj" "lira" "reference")
categories=("threshold" "fpr")
subcategories=("common_tp")

# For same attack different signals
#datasets=("cifar10")
#archs=("resnet56")
#mias=("shokri" "top_1_shokri" "top_3_shokri")
#categories=("threshold" "fpr")
#subcategories=("common_tp")

## For different distributions
#datasets=("cifar10" "cinic10")
#archs=("resnet56")
#mias=("shokri" "yeom")
#categories=("threshold" "fpr")
#subcategories=("common_tp")

option=("TPR")
seeds=(0 1 2 3 4 5)
fprs=(0.01)

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

experiment_dir="${DATA_DIR}/miae_experiment_aug_more_target_data"
graph_dir="$experiment_dir/graphs"
mkdir -p "$graph_dir"
if [ -d "$graph_dir" ]; then
    echo "Successfully created directory '$graph_dir'."
else
    echo "Error: Failed to create directory '$graph_dir'."
    exit 1
fi

upset_dir="$graph_dir/upset"
mkdir -p "$upset_dir"
if [ -d "$upset_dir" ]; then
    echo "Successfully created directory '$upset_dir'."
else
    echo "Error: Failed to create directory '$upset_dir'."
    exit 1
fi

for category in "${categories[@]}"; do
    mkdir -p "$upset_dir/$category"
    if [ -d "$upset_dir/$category" ]; then
        echo "Successfully created directory '$upset_dir/$category'."
    else
        echo "Error: Failed to create directory '$upset_dir/$category'."
        exit 1
    fi
done


# Generate the upset plots for the MIAE experiment
for category in "${categories[@]}"; do
    if [ "$category" == "threshold" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for subcategory in "${subcategories[@]}"; do
                    for opt in "${option[@]}"; do
                        threshold=0.5
                        plot_dir="$upset_dir/$category/common_tp/$dataset/$arch/threshold_${threshold}"
                        rm -rf "$plot_dir"
                        mkdir -p "$plot_dir"
                        graph_goal="common_tp"
                        graph_title="Upset for $dataset, $arch, common_tp"
                        graph_path="${plot_dir}"

                        python obtain_graphs.py --dataset "$dataset" \
                                                --architecture "$arch" \
                                                --attacks ${mialist} \
                                                --data_path "$experiment_dir" \
                                                --threshold "$threshold" \
                                                --FPR "0" \
                                                --graph_type "upset" \
                                                --graph_goal "$graph_goal" \
                                                --graph_title "$graph_title" \
                                                --graph_path "$graph_path" \
                                                --seed ${seedlist} \
                                                --opt ${opt}
                    done
                done
            done
        done
    elif [ "$category" == "fpr" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for subcategory in "${subcategories[@]}"; do
                    for opt in "${option[@]}"; do
                        for fpr in ${fprlist}; do
                            plot_dir="$upset_dir/$category/common_tp/$dataset/$arch/$opt/fpr_${fpr}"
                            rm -rf "$plot_dir"
                            mkdir -p "$plot_dir"
                            graph_goal="common_tp"
                            graph_title="Upset for $dataset, $arch, common_tp"
                            threshold=0
                            graph_path="${plot_dir}"

                            python obtain_graphs.py --dataset "$dataset" \
                                                    --architecture "$arch" \
                                                    --attacks ${mialist} \
                                                    --data_path "$experiment_dir" \
                                                    --threshold "$threshold" \
                                                    --FPR "$fpr" \
                                                    --graph_type "upset" \
                                                    --graph_goal "$graph_goal" \
                                                    --graph_title "$graph_title" \
                                                    --graph_path "$graph_path" \
                                                    --seed ${seedlist} \
                                                    --opt ${opt}
                        done
                    done
                done
            done
        done
    fi
done