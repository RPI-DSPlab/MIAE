import os
import torch
import torchvision.transforms as T

from mia import sample_metrics, attacks, eval_methods
from mia.sample_metrics import sample_metrics_config
from mia.utils import datasets, models
from utils.datasets.loader import load_dataset

dataset_dir = "datasets"
model_dir = "models"
metric_save_dir = "metrics"
result_dir = "results"
configs_dir = "configs"
dataset_name = "cifar10"


def main():
    for d in [dataset_dir, model_dir, result_dir, configs_dir, metric_save_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    """Example hardness metrics"""
    # loading the config for the sample metrics
    cs_config = sample_metrics_config.ConsistencyScoreConfig()
    il_config = sample_metrics_config.IterationLearnedConfig()
    pd_config = sample_metrics_config.PredictionDepthConfig()

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
    dataset = load_dataset(dataset_name, dataset_dir, train_transform, test_transform)

    # define model
    model = models.VGG16()

    consistency_score = sample_metrics.CSHardness(cs_config, model, dataset)
    iteration_learned = sample_metrics.IlHardness(il_config, model, dataset)
    prediction_depth = sample_metrics.PdHardness(pd_config, model, dataset)

    # load metrics if there's saved files in the result directory
    if len(os.listdir(metric_save_dir)) != 0:
        cs_metric_fn = [fn for fn in os.listdir(metric_save_dir + "/cs_results") if fn.startswith("cs")][0]
        consistency_score.load_metric(metric_save_dir + "/cs_results" + cs_metric_fn)
        iteration_learned.load_metric(metric_save_dir + "/il_results" + "result_avg")
        prediction_depth.load_metric(metric_save_dir + "/pd_results " + "result_avg")
    else:
        # train the metrics
        consistency_score.train_metric()
        iteration_learned.train_metric()
        prediction_depth.train_metric()


if __name__ == "__main__":
    main()
