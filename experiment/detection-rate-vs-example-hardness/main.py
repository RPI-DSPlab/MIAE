import os
import torch
import torchvision.transforms as T

import sys
sys.path.append('/home/wangz56/MIAE/mia')
sys.path.append('/home/wangz56/MIAE')
from mia import sample_metrics, attacks, visualization
from mia.sample_metrics import sample_metrics_config
from mia.utils import datasets, models
from utils.datasets.loader import load_dataset
from torchvision.models import vgg16

dataset_dir = "datasets"
model_dir = "models"
metric_save_dir = "metrics"
result_dir = "results"
configs_dir = "configs"
dataset_name = "cifar10"


def main(testing=False):
    # testing is a flag to indicate we are testing the code with fewer epochs

    for d in [dataset_dir, model_dir, result_dir, configs_dir, metric_save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    """Example hardness metrics"""
    # loading the config for the sample metrics
    cs_config = sample_metrics_config.ConsistencyScoreConfig()
    il_config = sample_metrics_config.IterationLearnedConfig()
    pd_config = sample_metrics_config.PredictionDepthConfig()

    if testing:
        cs_config.update({"num_epochs": 1, "n_runs": 1})
        il_config.update({"num_epochs": 1})
        pd_config.update({"num_epochs": 1})

    if len(os.listdir(configs_dir)) != 0:  # if the config files are not empty, load the config files
        fns = os.listdir(configs_dir)
        cs_config_fn = [fn for fn in fns if fn.startswith("consistency_score")][0]
        il_config_fn = [fn for fn in fns if fn.startswith("iteration_learned")][0]
        pd_config_fn = [fn for fn in fns if fn.startswith("prediction_depth")][0]
        cs_config.load(os.path.join(configs_dir, cs_config_fn))
        il_config.load(os.path.join(configs_dir, il_config_fn))
        pd_config.load(os.path.join(configs_dir, pd_config_fn))
    # update the save path
    cs_config.update({"save_path": metric_save_dir + "/cs_results"})
    il_config.update({"save_path": metric_save_dir + "/il_results"})

    pd_config.update({"save_path": metric_save_dir + "/pd_results"})

    """dataset and model"""
    # define transforms
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                 ])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
                                ])
    dataset = load_dataset(dataset_name,
                           dataset_dir,
                           train_transform=train_transform,
                           test_transform=test_transform,
                           target_transform=None)

    # define model
    ecd = vgg16().features
    model = models.VGG(ecd, 10)

    consistency_score = sample_metrics.CSHardness(cs_config, model, dataset)
    iteration_learned = sample_metrics.IlHardness(il_config, model, dataset)
    prediction_depth = sample_metrics.PdHardness(pd_config, model, dataset)

    # load metrics if there's saved files in the result directory
    if len(os.listdir(metric_save_dir)) != 0:

        if len([fn for fn in os.listdir(metric_save_dir + "/cs_results") if fn.startswith("cs")]) > 0:
            cs_metric_fn = [fn for fn in os.listdir(metric_save_dir + "/cs_results") if fn.startswith("cs")][0]
            consistency_score.load_metric(metric_save_dir + "/cs_results/" + cs_metric_fn)
        if len(os.listdir(metric_save_dir + "/il_results" + "/results_avg")) > 0:
            iteration_learned.load_metric(metric_save_dir + "/il_results" + "/results_avg")
        if len(os.listdir(metric_save_dir + "/pd_results" + "/results_avg")) > 0:
            prediction_depth.load_metric(metric_save_dir + "/pd_results " + "/results_avg")

    consistency_score.train_metric() if not consistency_score.ready else None
    iteration_learned.train_metric() if not iteration_learned.ready else None
    prediction_depth.train_metric() if not prediction_depth.ready else None


if __name__ == "__main__":
    main(testing=False)
