# This code implements "Membership Inference Attacks From First Principles", S&P 2022
# The code is based on the code from
# https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021


from base import ModelAccessType, AuxiliaryInfo, ModelAccess, MiAttack
# from attack_classifier import AttackClassifier # no classifer needed for LIRA

class LiraAuxiliaryInfo(AuxiliaryInfo):
    """
    Implementation of AuxiliaryInfo for Lira.
    """

    def __init__(self, your, parameters):
        """
        Initialize LiraAuxiliaryInfo.
        """
        # Your code here

class LiraModelAccess(ModelAccess):
    """
    Your implementation of ModelAccess for Lira.
    """

    def __init__(self, model, access_type: ModelAccessType):
        """
        Initialize LiraModelAccess.
        """
        super().__init__(model, access_type)

    def get_signal(self, data):
        """
        Implement the get_signal method.
        """
        # Your code here

class LiraMiAttack(MiAttack):
    """
    Implementation of MiAttack for Lira.
    """

    def __init__(self, target_model_access: LiraModelAccess, auxiliary_info: LiraAuxiliaryInfo, target_data=None):
        """
        Initialize LiraMiAttack.
        """
        super().__init__(target_model_access, auxiliary_info, target_data)

    def prepare(self, attack_config: dict):
        """
        Implement the prepare method.
        """
        # Your code here

    def infer(self, target_data):
        """
        Implement the infer method.
        """
        # Your code here
