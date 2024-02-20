# This code implements "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine
# Learning Models", NDSS 2019 The code is based on the code from
# https://github.com/AhmedSalem2/ML-Leaks
# https://github.com/Lab41/cyphercat

import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve
from tqdm import tqdm

from mia.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from mia.utils import datasets
from mia.utils import models
from mia.utils.set_seed import set_seed
from mia.utils.datasets import AbstractGeneralDataset

