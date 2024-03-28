# This script generates the accuracy for the MIAE experiment

# Get the datasets, architectures, MIAs and categories
datasets=("cifar10" "cifar100")
archs=("resnet56" "wrn32_4" "vgg16" "mobilenet")
mias=("losstraj" "shokri" "yeom")
categories=("threshold" "single_attack" "fpr")
subcategories=("common_tp" "pairwise")
seeds=(0 1 2 3)
fprs=(0.001 0.1 0.5 0.8)