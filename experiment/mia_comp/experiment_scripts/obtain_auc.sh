#experiment_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data'
#plot_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data/graphs/auc'
experiment_dir="/home/public/comp_mia_data/repeat_exp_set/miae_experiment_aug_more_target_data_3"
tmp_dir="/home/public/comp_mia_data"
plot_dir="$tmp_dir/repeat_graphs/auc3"

datasets=("cifar10")
archs=("resnet56")
mias=("losstraj" "shokri" "yeom" "lira" "aug")
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

        # plot the graphs
        graph_title="auc for ${dataset} ${arch} in log scale"
        graph_path="${plot_dir}/${dataset}/${arch}/auc_log_scale"
        python3 obtain_graphs.py --graph_type "auc"\
                                  --dataset "${dataset}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}\
                                  --log_scale "True"

        graph_title="auc for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/auc_linear_scale"
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