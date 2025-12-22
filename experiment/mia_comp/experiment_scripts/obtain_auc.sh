# modify this to set up directory:
DATA_DIR="data"


experiment_dir='${DATA_DIR}/miae_experiment_aug_more_target_data'
plot_dir='${DATA_DIR}/miae_experiment_aug_more_target_data/graphs_eli/auc'
tmp_dir='${DATA_DIR}'
#plot_dir="$tmp_dir/repeat_graphs/auc"

datasets=("cifar10")
archs=("resnet56")
mias=("losstraj" "shokri" "yeom" "aug" "calibration" "lira" "reference")
fprs=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)
seeds=(0 1 2 3 4 5)

# prepare the list of mias and fprs as arguments
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
done
fprlist=""
for fpr in "${fprs[@]}"; do
    fprlist+="${fpr} "
done
seedlist=""
for seed in "${seeds[@]}"; do
    seedlist+="${seed} "
done

for dataset in "${datasets[@]}"; do
    for arch in "${archs[@]}"; do
        # clean the plot directory
        rm -rf "${plot_dir:?}/${dataset:?}/${arch:?}"
        mkdir -p ${plot_dir}/${dataset}/${arch}

        # convert fprlist to space-separated string
        fprlist=$(printf "%s " "${fprs[@]}")

        graph_title="auc for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/auc"
        python3 obtain_graphs.py --graph_type "auc"\
                                  --dataset "${dataset}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}\
                                  --log_scale "False"
    done
done