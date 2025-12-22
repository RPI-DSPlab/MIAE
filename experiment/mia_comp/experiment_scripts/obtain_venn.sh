# This script generates Venn diagrams for the MIAE experiment under different settings

# modify this to set up directory:
DATA_DIR="data"

# ---------- Experiment Parameters ----------
# |     uncomment the one you want to run   |
# -------------------------------------------
# Define experiment parameters set 1 for standard MIAE
#datasets=("cifar100" "cinic10" "cifar10") # "cifar100" "cinic10" "cifar10"
#archs=("resnet56" "mobilenet" "wrn32_4" "vgg16") #"mobilenet" "wrn32_4" "vgg16"
#mias=("losstraj" "reference" "shokri" "yeom" "calibration" "aug" "lira") # "losstraj" "reference" "shokri" "yeom" "calibration" "aug" "lira"
#categories=("fpr" "threshold" "single_attack") # "threshold" "fpr" "single_attack"
#subcategories=("pairwise") # "common_tp"
#top_k=0
#experiment_dir="${DATA_DIR}/mia_standard_exp"
#option=("TPR")
#seeds=(0 1 2 3 4)
#fprs=(0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)

# Define experiment parameters set 2 for standard MIAE
#datasets=("purchase100" "texas100") # "purchase100" "texas100"
#archs=("mlp_for_texas_purchase")
#mias=("losstraj" "reference" "shokri" "yeom" "calibration" "lira")
#categories=("fpr" "threshold" "single_attack")
#subcategories=("pairwise")
#top_k=0
#experiment_dir="${DATA_DIR}/mia_standard_exp"
#option=("TPR")
#seeds=(0 1 2 3 4)
#fprs=(0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)

#Define experiment parameters set 3 for different distributions
#datasets=("cifar10" "cinic10")
#archs=("resnet56")
#mias=("shokri" "yeom")
#categories=("dif_distribution")
#top_k=0
#experiment_dir="${DATA_DIR}/same_attack_different_signal"
#option=("TPR")
#seeds=(0 1 2)
##fprs=(0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)
#fprs=(0.01 0.1)

# # Define experiment parameters set 4 for same attack different signal
#datasets=("cifar10")
#archs=("resnet56")
#mias=("shokri_top_1" "shokri_top_3" "shokri_top_10")
#categories=("fpr")
#subcategories=("common_tp" "pairwise")
#top_k=1
#experiment_dir="${DATA_DIR}/top_k_shokri_new"
#option=("TPR")
#seeds=(0 1 2)
#fprs=(0.01 0.1)

# Define experiment parameters set comparing LiRA online vs offline
#datasets=("cifar10")
#archs=("resnet56")
#mias=("lira" "lira_offline")
#categories=("fpr")
#subcategories=("common_tp" "pairwise")
#experiment_dir="${DATA_DIR}/miae_standard_exp"
#option=("TPR")
#top_k=0 # means we are not consider any top k variation
#seeds=(0 1 2 3 4 5)
#fprs=(0.01 0.1)


# ------------------------------
# |       actual scripts       |
# |       do not modify        |
# ------------------------------

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

datasetlist=""
for dataset in "${datasets[@]}"; do
    datasetlist+="${dataset} "
done


graph_dir="$experiment_dir/venn_lira_online_vs_offline"
mkdir -p "$graph_dir"

if [ -d "$graph_dir" ]; then
    echo "Successfully created directory '$graph_dir'."
else
    echo "Error: Failed to create directory '$graph_dir'."
    exit 1
fi

venn_dir="$graph_dir/venn"
mkdir -p "$venn_dir"
if [ -d "$venn_dir" ]; then
    echo "Successfully created directory '$venn_dir'."
else
    echo "Error: Failed to create directory '$venn_dir'."
    exit 1
fi

for category in "${categories[@]}"; do
    mkdir -p "$venn_dir/$category"
    if [ -d "$venn_dir/$category" ]; then
        echo "Successfully created directory '$venn_dir/$category'."
    else
        echo "Error: Failed to create directory '$venn_dir/$category'."
        exit 1
    fi
done


# Main loop: Generate Venn diagrams for each experimental category
for category in "${categories[@]}"; do
    if [ "$category" == "threshold" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                echo "Running threshold on $dataset, $arch"
                for subcategory in "${subcategories[@]}"; do
                    for opt in "${option[@]}"; do
                        threshold=0.5
                        if [ "$subcategory" == "common_tp" ]; then
                            plot_dir="$venn_dir/$category/common_tp/$dataset/$arch/threshold_${threshold}"
                            rm -rf "$plot_dir"
                            mkdir -p "$plot_dir"
                            graph_goal="common_tp"
                            graph_title="Venn for $dataset, $arch, common_tp"
                        elif [ "$subcategory" == "pairwise" ]; then
                            plot_dir="$venn_dir/$category/pairwise/$dataset/$arch/threshold_${threshold}"
                            rm -rf "$plot_dir"
                            mkdir -p "$plot_dir"
                            graph_goal="pairwise"
                            graph_title="$dataset, $arch, pairwise"
                        fi

                        graph_path="${plot_dir}"

                        python obtain_graphs.py --dataset "$dataset" \
                                                --architecture "$arch" \
                                                --attacks ${mialist} \
                                                --data_path "$experiment_dir" \
                                                --threshold "$threshold" \
                                                --FPR "0" \
                                                --graph_type "venn" \
                                                --graph_goal "$graph_goal" \
                                                --graph_title "$graph_title" \
                                                --graph_path "$graph_path" \
                                                --seed ${seedlist} \
                                                --opt ${opt} \
                                                --top_k ${top_k}
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
                            echo "Running fpr on $dataset, $arch on fpr = $fpr"
                            if [ "$subcategory" == "common_tp" ]; then
                                plot_dir="$venn_dir/$category/common_tp/$dataset/$arch/$opt/fpr_${fpr}"
                                rm -rf "$plot_dir"
                                mkdir -p "$plot_dir"
                                graph_goal="common_tp"
                                graph_title="Venn for $dataset, $arch, common_tp"
                            elif [ "$subcategory" == "pairwise" ]; then
                                plot_dir="$venn_dir/$category/pairwise/$dataset/$arch/$opt/fpr_${fpr}"
                                rm -rf "$plot_dir"
                                mkdir -p "$plot_dir"
                                graph_goal="pairwise"
                                graph_title="$dataset, $arch, pairwise"
                            fi

                            threshold=0
                            graph_path="${plot_dir}"

                            python obtain_graphs.py --dataset "$dataset" \
                                                    --architecture "$arch" \
                                                    --attacks ${mialist} \
                                                    --data_path "$experiment_dir" \
                                                    --threshold "0" \
                                                    --FPR "$fpr" \
                                                    --graph_type "venn" \
                                                    --graph_goal "$graph_goal" \
                                                    --graph_title "$graph_title" \
                                                    --graph_path "$graph_path" \
                                                    --seed ${seedlist} \
                                                    --opt ${opt} \
                                                    --top_k ${top_k}
                        done
                    done
                done
            done
        done
    elif [ "$category" == "single_attack" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for mia in "${mias[@]}"; do
                    echo "Running single_attack on $dataset, $arch, $mia"
                    for opt in "${option[@]}"; do
                        for fpr in ${fprlist}; do
                            plot_dir="$venn_dir/$category/$dataset/$arch/$opt/$mia/fpr_$fpr"
                            rm -rf "$plot_dir"
                            mkdir -p "$plot_dir"

                            # run the experiment
                            graph_title="$dataset, $arch, $mia (FPR: $fpr)"
                            graph_path="${plot_dir}"

                            python obtain_graphs.py --dataset "$dataset" \
                                                    --architecture "$arch" \
                                                    --attacks ${mialist} \
                                                    --data_path "$experiment_dir" \
                                                    --single_attack_name "$mia" \
                                                    --threshold "0" \
                                                    --FPR $fpr \
                                                    --graph_type "venn" \
                                                    --graph_goal "single_attack" \
                                                    --graph_title "$graph_title" \
                                                    --graph_path "$graph_path" \
                                                    --seed ${seedlist} \
                                                    --opt ${opt}  \
                                                    --top_k ${top_k}
                        done
                    done
                done
            done
        done
    elif [ "$category" == "dif_distribution" ]; then
        for arch in "${archs[@]}"; do
            for mia in "${mias[@]}"; do
                for opt in "${option[@]}"; do
                    for fpr in ${fprlist}; do
                        plot_dir="$venn_dir/$category/$arch/$opt/$mia/fpr_$fpr"
                        rm -rf "$plot_dir"
                        mkdir -p "$plot_dir"

                        graph_title="$dataset, $arch, $mia (FPR: $fpr)"
                        graph_path="${plot_dir}"

                        python obtain_graphs.py --dataset "-" \
                                                --architecture "$arch" \
                                                --attacks ${mialist} \
                                                --data_path "$experiment_dir" \
                                                --single_attack_name "$mia" \
                                                --threshold "0" \
                                                --FPR $fpr \
                                                --graph_type "venn" \
                                                --graph_goal "dif_distribution" \
                                                --graph_title "$graph_title" \
                                                --graph_path "$graph_path" \
                                                --seed ${seedlist} \
                                                --dataset_list ${datasetlist} \
                                                --opt ${opt} \
                                                --top_k ${top_k}
                    done
                done
            done
        done
    elif [ "$category" == "model_compare" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for fpr in ${fprlist}; do
                    plot_dir="$venn_dir/$category/$dataset/fpr_$fpr/$arch"
                    rm -rf "$plot_dir"
                    mkdir -p "$plot_dir"

                    # run the experiment
                    graph_title="$dataset, $arch, $mia (FPR: $fpr)"
                    graph_path="${plot_dir}"

                    python obtain_graphs.py --dataset "$dataset" \
                                            --architecture "$arch" \
                                            --attacks ${mialist} \
                                            --data_path "$experiment_dir" \
                                            --single_attack_name "" \
                                            --threshold "0" \
                                            --FPR $fpr \
                                            --graph_type "venn" \
                                            --graph_goal "model_compare" \
                                            --graph_title "$graph_title" \
                                            --graph_path "$graph_path" \
                                            --seed ${seedlist} \
                                            --opt "" \
                                            --top_k ${top_k}
                done
            done
        done
    fi
done
