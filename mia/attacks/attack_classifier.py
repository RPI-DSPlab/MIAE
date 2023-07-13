# A set of typical model candidates that can be trained for the inference attacks, including  KNN< randome forest, neural network,
# logistic regression, decision tree, etc. This class is modified from the code from
#     tensorflow_privacy/privacy/privacy_tests/membership_inference_attack/sample_metrics_models.py
from abc import ABC, abstractmethod
import contextlib
import logging
from sklearn import ensemble
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn.utils import parallel_backend
import numpy as np
from enum import Enum

class AttackType(Enum):
  """An enum define attack types."""
  LOGISTIC_REGRESSION = 'lr'
  MULTI_LAYERED_PERCEPTRON = 'mlp'
  RANDOM_FOREST = 'rf'
  K_NEAREST_NEIGHBORS = 'knn'
  THRESHOLD_ATTACK = 'threshold'
  THRESHOLD_ENTROPY_ATTACK = 'threshold-entropy'


class AttackClassifier(ABC):
    """
    Base class for all attack classifier implementations that are used by MIA attacks, which can be including  Threshold,
    KNN， random forest, neural network,logistic regression, decision tree, etc.
    """

    def __init__(self, config: dict):
        self.model = None
        self.backend = config.get('backend')
        self.n_jobs = config.get('n_jobs', -1)
        if self.backend is None:
            self.ctx_mgr = contextlib.nullcontext()
            self.n_jobs = 1
            logging.info('Using single-threaded backend for training.')
        else:
            self.ctx_mgr = parallel_backend(self.backend, n_jobs=self.n_jobs)
            logging.info('Using %s backend for training.', self.backend)

    @abstractmethod
    def build_classifier(self, input_signals, member_labels, sample_weight=None):
        """
        Build the attack model.
        :param input_features: the input features.
        :param member_labels: a vector of booleans indicating if a sample is a member of training dataset or not.
        1-member 0-non-member.
        :param sample_weight: the sample weight.
        :return: the attack model. If it is a threshold based classifier, it returns a threshold function.
        """
        pass

    def threshold_func(self, input_signals):
        pass
    @abstractmethod
    def predict(self, input_signals):
        """
        Predict the membership of the input features.
        :param input_features: the input features.
        :return: a vector of binary or probabilities denoting whether an example belongs to the training dataset.
        """
        if self.model is None:
            raise ValueError('Model is not trained yet.')
        elif self.model is "Threshold":
            return self.model(input_signals)
        return self.model.predict_proba(input_signals)[:, 1]



# To do, add more attack classifiers here like threshold based classifier.

class LrAC(AttackClassifier):
    """Logistic regression attack classifier."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.param_grid = config.get('param_grid', {'C': np.logspace(-4, 2, 10)})

    def build_classifier(self, input_signals, member_labels, sample_weight=None):
        with self.ctx_mgr:
            lr = linear_model.LogisticRegression(solver='lbfgs', n_jobs=self.n_jobs)
            param_grid = self.param_grid
            if param_grid is not None:
                model = model_selection.GridSearchCV(
                    lr, param_grid=self.param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
            else:
                model = lr
            model.fit(input_signals, member_labels, sample_weight=sample_weight)
        self.model = model


class MlpAC(AttackClassifier):
    """Multilayer perceptron attacker."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.param_grid = config.get('param_grid', {
                'hidden_layer_sizes': [(64,), (32, 32)],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001, 0.01],
            })
    def build_classifier(self, input_signals, member_labels, sample_weight=None):
        del sample_weight  # MLP attacker does not use sample weights.
        with self.ctx_mgr:
            mlp_model = neural_network.MLPClassifier()
            model = model_selection.GridSearchCV(
                mlp_model, param_grid=self.param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
            model.fit(input_signals, member_labels)
        self.model = model


class RandomForestAC(AttackClassifier):
    """Random forest attacker."""

    def __init__(self, config):
        super().__init__(config)
        self.param_grid = config.get('param_grid', {
                'n_estimators': [100],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            })
    def build_classifier(self, input_signals, member_labels, sample_weight=None):
        """Setup a random forest pipeline with cross-validation."""
        with self.ctx_mgr:
            rf_model = ensemble.RandomForestClassifier(n_jobs=self.n_jobs)

            model = model_selection.GridSearchCV(
                rf_model, param_grid=self.param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
            model.fit(input_signals, member_labels, sample_weight=sample_weight)
        self.model = model


class KnnAC(AttackClassifier):
    """K nearest neighbor attack model."""

    def __init__(self, config):
        super().__init__(config)
        self.param_grid = config.get('param_grid', {
                'n_neighbors': [3, 5, 7],
            })

    def build_classifier(self, input_signals, member_labels, sample_weight=None):
        del sample_weight  # K-NN attacker does not use sample weights.
        with self.ctx_mgr:
            knn_model = neighbors.KNeighborsClassifier(n_jobs=self.n_jobs)
            model = model_selection.GridSearchCV(
                knn_model, param_grid=self.param_grid, cv=3, n_jobs=self.n_jobs, verbose=0)
            model.fit(input_signals, member_labels)
        self.model = model
