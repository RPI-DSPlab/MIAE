experiment_dir='/data/public/miae_experiment_aug'

plot_dir='/data/public/miae_experiment_aug/graphs/auc'

datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
#mias=("losstraj" "shokri" "yeom" "lira")
mias=("losstraj" "shokri" "yeom")
fprs=(0.001 0.5 0.8)
seeds=(0 1 2 3)

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