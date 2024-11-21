experiment_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data'

plot_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data/graphs/single_seed_ensemble'

cd /home/wangz56/MIAE_training_dir/MIAE/experiment/mia_comp

datasets=("cifar10")
archs=("resnet56")
#mias=("losstraj" "shokri" "yeom" "lira")
mias=("losstraj" "shokri" "yeom" "lira" "aug")
seeds=(0 1 2 3 4 5)
#seeds=(0 1 2 3 4 5 6 7 8 9 10 11)

# prepare the list of mias and fprs as arguments
mialist=""
for mia in "${mias[@]}"; do
    mialist+="${mia} "
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

        # plot the graphs
        # common TP (intersection of all seeds)
        graph_title="single_seed_ensemble ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/single_attack_seed_ensemble"
        python3 obtain_graphs.py --graph_type "single_attack_seed_ensemble"\
                                  --dataset "${dataset}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --skip 2\
                                  --seed ${seedlist}
    done
done