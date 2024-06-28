# This script generates Venn diagrams for the MIAE experiment
datasets=("cifar10" "cinic10")
archs=("resnet56")
mias=("shokri")
categories=("single_attack" "dif_distribution")
subcategories=("pairwise" "common_tp")
option=("TPR")
seeds=(0 1 2)
fprs=(0.001 0.01 0 0.1 0.2 0.3 0.4 0.5 0.8)

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

datasetlist=""
for dataset in "${datasets[@]}"; do
    datasetlist+="${dataset} "
done


experiment_dir="/data/public/comp_mia_data/same_attack_different_distribution"
graph_dir="$experiment_dir/graphs"
mkdir -p "$graph_dir"

# Check if directory creation was successful
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

cd /home/zhangc26/MIAE/experiment/mia_comp

# Generate Venn diagrams for the MIAE experiment when the goal is common_tp
for category in "${categories[@]}"; do
    if [ "$category" == "threshold" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
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
                                                    --threshold "$threshold" \
                                                    --FPR "$fpr" \
                                                    --graph_type "venn" \
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
    elif [ "$category" == "single_attack" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for mia in "${mias[@]}"; do
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
                                                    --opt ${opt}
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
                                                --opt ${opt}
                    done
                done
            done
        done
    fi
done