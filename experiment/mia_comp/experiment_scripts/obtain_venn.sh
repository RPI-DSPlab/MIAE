# This script generates Venn diagrams for the MIAE experiment

# Get the datasets, architectures, MIAs and categories
datasets=("cifar10")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri" "yeom")
categories=("threshold/single_attack" "threshold/common_TP" "FPR/single_attack" "FPR/common_TP")
seeds=(0 1 2 3)

# Prepare the parameter lists for the experiment
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
done

seedlist=""
for seed in "${seeds[@]}"; do
    seedlist+="${seed} "
done


experiment_dir="/data/public/miae_experiment_aug"
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
    echo "Directory '$graph_dir' already exists. Skipping..."
else
    echo "Error: Failed to create directory '$venn_dir'."
    exit 1
fi

for category in "${categories[@]}"; do
    mkdir -p "$venn_dir/$category"
done

cd /home/zhangc26/MIAE/experiment/mia_comp

conda activate rpidsp

for category in "${categories[@]}"; do
    IFS='/' read -r -a category_parts <<< "$category"
    sub_dir="${category_parts[0]}"
    sub_sub_dir="${category_parts[1]}"
    if [ "$sub_sub_dir" = "common_TP" ]; then
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                miae_dir="$graph_dir/venn/$category/$dataset/$arch"
                mkdir -p "$miae_dir"

                # Generate Venn diagrams with mandatory and optional arguments
                python obtain_graphs.py --dataset "$dataset" \
                                        --architecture "$arch" \
                                        --attacks ${mialist}\
                                        --graph_type "venn" \
                                        --graph_title "Venn Diagram for $dataset, $arch" \
                                        --graph_path "$miae_dir" \
                                        --seed ${seedlist}\
                                        --data_aug "True"

                # Check if the Venn diagram was created
#                if [ -f "$miae_dir/Venn Diagram for $dataset, $arch.png" ]; then
#                    echo "Venn diagram for $dataset, $arch successfully created."
#                else
#                    echo "Error: Failed to create Venn diagram for $dataset, $arch."
#                    exit 1
#                fi
            done
        done
    else
        for dataset in "${datasets[@]}"; do
            for arch in "${archs[@]}"; do
                for mia in "${mias[@]}"; do
                    miae_dir="$graph_dir/venn/$category/$dataset/$arch/$mia"
                    mkdir -p "$miae_dir"

                    # Generate Venn diagrams with mandatory and optional arguments
                    python obtain_graphs.py --dataset "$dataset" \
                                            --architecture "$arch" \
                                            --attacks ${mialist}\
                                            --graph_type "venn" \
                                            --graph_title "Venn Diagram for $dataset, $arch, $mia" \
                                            --graph_path "$miae_dir" \
                                            --seed ${seedlist}\
                                            --data_aug "True"

                    # Check if the Venn diagram was created
#                    if [ -f "$miae_dir/Venn Diagram for $dataset, $arch, $mia.png" ]; then
#                        echo "Venn diagram for $dataset, $arch, $mia successfully created."
#                    else
#                        echo "Error: Failed to create Venn diagram for $dataset, $arch, $mia."
#                        exit 1
#                    fi
                done
            done
        done
    fi
done