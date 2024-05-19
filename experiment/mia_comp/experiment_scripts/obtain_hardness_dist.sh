# add the pwd's  ../.. to the python path
export PYTHONPATH=$(pwd)/../..

experiment_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data'

# il
# hardness_path='/data/public/comp_mia_data/example_hardness_aug'
# pd
#hardness_path='/data/public/comp_mia_data/example_hardness_aug/cifar10/vgg16/pd/pd_score.pkl'
# aleatoric
# hardness_path='/data/public/comp_mia_data/example_hardness_external/cifar10/aleatoric_uncertainty_score.pkl'
# epistemic
# hardness_path='/data/public/comp_mia_data/example_hardness_external/cifar10/epistemic_uncertainty_score.pkl'
# deepsvdd
# hardness_path='/data/public/comp_mia_data/example_hardness_external/cifar10/deepsvdd_score.pkl'
# gradient_norm
# hardness_path='/data/public/comp_mia_data/example_hardness_external/cifar10/gradient_norm_score.pkl'
# order
hardness_path='/data/public/comp_mia_data/example_hardness_external/cifar10/order_score.pkl'


plot_dir='/data/public/comp_mia_data/miae_experiment_aug_more_target_data/graphs/hardness_dist'


datasets=("cifar10")
archs=("resnet56" "wrn32_4")
mias=("losstraj" "shokri" "yeom" "aug" "lira")
fprs=(0.001 0.1 0.8)
seeds=(0 1 2 3)

hardness=("order")

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
        rm -rf ${plot_dir}/${dataset}/${arch}/${eh}
        mkdir -p ${plot_dir}/${dataset}/${arch}/${eh}

        # convert fprlist to space-separated string
        fprlist=$(printf "%s " "${fprs[@]}")

        # plot the graphs
        graph_title="${eh} distribution for ${dataset} ${arch}"
        graph_path="${plot_dir}/${dataset}/${arch}/${eh}/${eh}_hardness_distribution"
        python3 obtain_graphs.py --graph_type "hardness_distribution"\
                                  --dataset "${dataset}"\
                                  --hardness_path "${hardness_path}"\
                                  --graph_title "${graph_title}"\
                                  --data_path "${experiment_dir}"\
                                  --graph_path "${graph_path}"\
                                  --architecture "${arch}"\
                                  --attacks ${mialist}\
                                  --fpr ${fprlist}\
                                  --seed ${seedlist}\
                                  --hardness ${eh}\
                                  --external 1
                                  # when this is 1 true, we are using an external hardness metric
                                  # the path to hardness should be specified to .pkl file
        done
    done
done