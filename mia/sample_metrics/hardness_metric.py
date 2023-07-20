import time

from base import ExampleMetric
import sys
import torch
import os
import numpy as np
import json
sys.path.append("..")
from ..utils import models as smmodels
from ..utils import datasets as smdatasets

class IterHardness(ExampleMetric):
    def __init__(self):
        pass

    def compute_metric(self):
        pass
