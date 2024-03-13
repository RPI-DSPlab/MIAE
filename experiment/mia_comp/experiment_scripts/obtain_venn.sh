# This script is used to obtain the Venn Diagrams for different experiments
seed=0
if [ $# -eq 1 ]; then  # if the number of arguments is 1, the argument is the seed
    seed=$1
fi
echo "obtain_venn.sh seed = $seed"



# Get the datasets, architectures, MIAs and categories
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri" "yeom")
categories=("different_attacks_same_seed" "different_attacks_same_FPR" "single_attack_different_seeds")

experiment_dir="/data/public/miae_experiment_aug"
graph_dir="$experiment_dir/graphs"
mkdir -p "$graphs_dir"
# Check if directory creation was successful
if [ -d "$graphs_dir" ]; then
    echo "Directory '$graphs_dir' successfully created."
else
    echo "Error: Failed to create directory '$graphs_dir'."
    exit 1
fi


venns_dir="$graphs_dir/venns"
mkdir -p "$venns_dir"
if [ -d "$venns_dir" ]; then
    echo "Directory '$venns_dir' successfully created."
else
    echo "Error: Failed to create directory '$venns_dir'."
    exit 1
fi

for category in "${categories[@]}"; do
    mkdir -p "$venns_dir/$category"
done

cd /home/zhangc26/MIAE/experiment/mia_comp

conda activate rpidsp

for category in "${categories[@]}"; do
    for dataset in "${datasets[@]}"; do
        for arch in "${archs[@]}"; do
            mkdir -p "$venns_dir/$category/$dataset/$arch"
            for mia in "${mias[@]}"; do
                miae_dir="$venns_dir/$category/$dataset/$arch/$mia"
                if [ -d "$miae_dir" ]; then
                    echo "Directory '$miae_dir' already exists. Skipping..."
                    continue
                fi
                mkdir -p "$miae_dir"

                # Generate Venn diagrams with mandatory and optional arguments
                python obtain_venn.py --dataset "$dataset" \
                                      --architecture "$arch" \
                                      --attacks "$mia" \
                                      --graph-type "venn" \
                                      --graph-title "Venn Diagram for $dataset, $arch, $mia" \
                                      --graph-path "$miae_dir" \
                                      --seed "$seed" \
                                      --data_aug "True"
            done
        done
    done
done
echo "Venn diagrams generated successfully."