experiment_dir='/data/public/miae_experiment_aug'

hardness_path='/data/public/example_hardness_aug'

plot_dir='/data/public/miae_experiment_aug/graphs/hardness_dist'

datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
#mias=("losstraj" "shokri" "yeom" "lira")
mias=("losstraj" "shokri" "yeom")
fprs=(0.001 0.5 0.8)
seeds=(0 1 2 3)

hardness=("il")

# prepare the list of mias as arguments
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
      for eh in "${hardness[@]}"; do
        # clean the plot directory
        rm -rf ${plot_dir}/${dataset}/${arch}
        mkdir -p ${plot_dir}/${dataset}/${arch}

        # convert fprlist to space-separated string
        fprlist=$(printf "%s " "${fprs[@]}")

        # plot the graphs
        graph_title="${eh} distribution for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/hardness_distribution"
        python3 obtain_graphs.py --graph-type "hardness_distribution"\
                                  --dataset "${dataset}"\
                                  --hardness-path "${hardness_path}"\
                                  --graph-title "${graph_title}"\
                                  --data-path "${experiment_dir}"\
                                  --graph-path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}\
                                  --hardness ${eh}

        done
    done
done