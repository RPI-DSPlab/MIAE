experiment_dir='/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_1'

plot_dir='/data/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_1/graphs/auc'

datasets=("cifar10")
archs=("resnet56")
#mias=("losstraj" "shokri" "yeom" "lira")
mias=("losstraj" "yeom" "aug" "calibration" "lira")
fprs=()
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
                                  --seed ${seedlist}\
                                  --log_scale "False"
    done
done