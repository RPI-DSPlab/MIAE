"""
    Enum class for attacks. Also contains the base attack class.
"""
from enum import Enum
from miae.attacks.base import LLM_ModelAccess as Model


# Attack definitions
class AllAttacks(str, Enum):
    ZLIB = "zlib"
    MIN_K = "min_k"
    NEIGHBOR = "neighbor"
    LOSS = "loss"

# Base attack class
class Attack:
    def __init__(self, config, target_model: Model):
        self.confid = config
        self.target_model = target_model

    def _attack(self, document, probs, tokens=None):
        """
        Actual logic for attack.
        """
        raise NotImplementedError("Attack must implement attack()")
