# modify this to set up directory:
DATA_DIR="data"

experiment_dir="${DATA_DIR}/miae_standard_exp"

plot_dir="${DATA_DIR}/miae_standard_exp/multiseed_convergence"

datasets=("cifar10" "cifar100")
archs=("resnet56")
mias=("losstraj" "shokri" "yeom" "lira" "aug" "calibration" "reference")
fprs=(0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.8)
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
#seeds=(0 1 2 3 4 5)

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
        echo "plots at ${plot_dir}/${dataset}/${arch}"

        # convert fprlist to space-separated string
        fprlist=$(printf "%s " "${fprs[@]}")

        # plot the graphs
        # common TP (intersection of all seeds)
        graph_title="multi-seed common TP for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/multi_seed_intersection_TP"
        rm -rf "${graph_path}"
        mkdir -p ${graph_path}
        python3 obtain_graphs.py --graph_type "multi_seed_convergence_intersection"\
                                  --dataset "${dataset}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}

        # attack coverage (union of all seeds)
        graph_title="multi-seed attack coverage for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/multi_seed_union_TP"
        rm -rf "${graph_path}"
        mkdir -p ${graph_path}
        python3 obtain_graphs.py --graph_type "multi_seed_convergence_union"\
                                  --dataset "${dataset}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}
    done
done