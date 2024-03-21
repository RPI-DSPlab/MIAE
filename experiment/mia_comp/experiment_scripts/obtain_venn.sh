# This script generates Venn diagrams for the MIAE experiment

# Get the datasets, architectures, MIAs and categories
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri" "yeom")
categories=("threshold" "single_attack" "fpr")
subcategories=("common_tp" "pairwise")
seeds=(0 1 2 3)
fprs=(0.001 0.5 0.8)

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


experiment_dir="/data/public/comp_mia_data/miae_experiment_aug"
graph_dir="$experiment_dir/graphs"
mkdir -p "$graph_dir"

# Check if directory creation was successful
if [ -d "$graph_dir" ]; then
    echo "Directory '$graph_dir' already exists. Skipping..."
else
    echo "Error: Failed to create directory '$graph_dir'."
    exit 1
fi

venn_dir="$graph_dir/venn"
mkdir -p "$venn_dir"
if [ -d "$venn_dir" ]; then
    echo "Directory '$venn_dir' already exists. Skipping..."
else
    echo "Error: Failed to create directory '$venn_dir'."
    exit 1
fi

for category in "${categories[@]}"; do
    mkdir -p "$venn_dir/$category"
    if [ -d "$venn_dir/$category" ]; then
        echo "Directory '$venn_dir/$category' already exists. Skipping..."
    else
        echo "Error: Failed to create directory '$venn_dir/$category'."
        exit 1
    fi
done

cd /home/zhangc26/MIAE/experiment/mia_comp

conda activate rpidsp

# Generate Venn diagrams for the MIAE experiment when the goal is common_tp
for category in "${categories[@]}"; do
    # if categroy is threshold or fpr, echo the title
    if [ "$category" == "threshold" ] || [ "$category" == "fpr" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for subcategory in "${subcategories[@]}"; do
                    if [ "$subcategory" == "common_tp" ]; then
                        plot_dir="$venn_dir/$category/common_tp/$dataset/$arch"
                        rm -rf "$plot_dir"
                        mkdir -p "$plot_dir"
                        graph_goal="common_tp"
                        graph_title="Venn for $dataset, $arch, common_tp"
                    elif [ "$subcategory" == "pairwise" ]; then
                        plot_dir="$venn_dir/$category/pairwise/$dataset/$arch"
                        rm -rf "$plot_dir"
                        mkdir -p "$plot_dir"
                        graph_goal="pairwise"
                        graph_title="Venn for $dataset, $arch, pairwise"
                    fi

                    if [ "$category" == "threshold" ]; then
                        threshold=0.5
                        graph_title=${graph_title}
                        graph_path="${plot_dir}/threshold"
                    elif [ "$category" == "fpr" ]; then
                        threshold=0
                        graph_title=${graph_title}
                        graph_path="${plot_dir}/fpr"
                    fi

                    python obtain_graphs.py --dataset "$dataset" \
                                                    --architecture "$arch" \
                                                    --attacks ${mialist} \
                                                    --dpata_path "$experiment_dir" \
                                                    --threshold "$threshold" \
                                                    --fpr ${fprlist}\
                                                    --graph_type "venn" \
                                                    --graph_goal "$graph_goal" \
                                                    --graph_title "$graph_title" \
                                                    --graph_path "$graph_path" \
                                                    --seed ${seedlist}
                done
            done
        done
    elif [ "$category" == "single_attack" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for mia in "${mias[@]}"; do
                    plot_dir="$venn_dir/$category/$dataset/$arch/$mia"
                    rm -rf "$plot_dir"
                    mkdir -p "$plot_dir"

                    # run the experiment
                    graph_title="Venn for $dataset, $arch, $mia"
                    graph_path="${plot_dir}/venn_${mia}"
                    fpr_tmp_list="0.0 0.0 0.0"
                    python obtain_graphs.py --dataset "$dataset" \
                                                    --architecture "$arch" \
                                                    --attacks ${mialist} \
                                                    --data_path "$experiment_dir" \
                                                    --single_attack_name "$mia" \
                                                    --threshold "0" \
                                                    --fpr ${fpr_tmp_list} \
                                                    --graph_type "venn" \
                                                    --graph_goal "single_attack" \
                                                    --graph_title "$graph_title" \
                                                    --graph_path "$graph_path" \
                                                    --seed ${seedlist}
                done
            done
        done
    fi
done