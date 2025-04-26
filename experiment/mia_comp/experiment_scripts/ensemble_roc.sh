# modify this to set up directory:
DATA_DIR="data"


data_dir="${DATA_DIR}/miae_standard_exp"

datasets="cifar10 cifar100 cinic10"
# datasets="texas100 purchase100"
archs="resnet56 wrn32_4 vgg16 mobilenet"
# archs="mlp_for_texas_purchase"
mias="lira reference losstraj calibration"
seeds="0 1 2 3 4 5"


python ensemble/ensemble_roc.py --path_to_data "$data_dir" \
                                --datasets ${datasets} \
                                --attack_list ${mias} \
                                --seeds ${seeds} \
                                --models ${archs} \
                                --num_fpr_for_table_ensemble 100

# datasets=("purchase100" "taxes100")
# archs="mlp_for_texas_purchase"
# mias="lira reference losstraj calibration"
# seeds="0 1 2 3 4 5"

# python ensemble/ensemble_roc.py --path_to_data "$data_dir" \
#                                 --datasets ${datasets} \
#                                 --attack_list ${mias} \
#                                 --seeds ${seeds} \
#                                 --models ${archs} \
#                                 --num_fpr_for_table_ensemble 100