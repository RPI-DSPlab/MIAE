import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
import pickle
from enum import Enum

class allAttacks(str, Enum):
    """
    An enum describing all the attacks that are possible
    """
    LOSS = "loss"
    NEIGHBOR = "neighbor"
    ZLIB = "zlib"

class BaseAttack:
    """
    Base class for all attacks
    """
