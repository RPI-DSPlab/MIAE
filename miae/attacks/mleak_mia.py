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

from miae.attacks.base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
from miae.utils import datasets
from miae.utils import models
from miae.utils.set_seed import set_seed
from miae.utils.datasets import AbstractGeneralDataset

