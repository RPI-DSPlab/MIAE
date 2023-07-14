import json

from sample_metrics.outlier.methods.deepSVDD import DeepSVDD
from sample_metrics.outlier.methods.oneClassSVM import OneClassSVM


class OutlierDetectionLoader:
    def __init__(self, model_type):
        self.model_type = model_type
        self.config = self.generate_default_config()

    def generate_default_config(self):
        if self.model_type == 'OneClassSVM':
            return {
                'nu': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale',
                'data': None  # The user should provide the data
            }
        elif self.model_type == 'DeepSVDD':
            return {
                "dataset_name": "cifar10",
                "net_name": "cifar10_LeNet",
                "load_model": False,
                "xp_path": "./",
                "data_path": "./data",
                "objective": "one-class",
                "nu": 0.1,
                "device": "cuda",
                "seed": -1,
                "optimizer_name": "adam",
                "lr": 0.001,
                "n_epochs": 50,
                "lr_milestone": [50],
                "batch_size": 128,
                "weight_decay": 1e-6,
                "pretrain": True,
                "ae_optimizer_name": "adam",
                "ae_lr": 0.001,
                "ae_n_epochs": 100,
                "ae_lr_milestone": [0],
                "ae_batch_size": 128,
                "ae_weight_decay": 1e-6,
                "n_jobs_dataloader": 0,
                "normal_class": 0
            }
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def update_config(self, updates):
        self.config.update(updates)

    def get_model(self):
        if self.model_type == 'OneClassSVM':
            return OneClassSVM(self.config)
        elif self.model_type == 'DeepSVDD':
            return DeepSVDD(self.config)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def save_config_to_file(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=4)
