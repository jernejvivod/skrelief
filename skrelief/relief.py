import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
import os

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import Relief as Relief_jl


class Relief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the Relief algorithm.

    Reference:
        Kenji Kira and Larry A. Rendell. The feature selection problem: Tra-
        ditional methods and a new algorithm. In Proceedings of the Tenth
        National Conference on Artificial Intelligence, AAAI’92, pages 129–134.
        AAAI Press, 1992.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        dist_func (function): function used to measure similarities between samples.
        _relief (function): function implementing Relief algorithm written in Julia programming language.
    """

    def __init__(self, n_features_to_select=10, m=-1, dist_func=lambda x1, x2: np.sum(np.abs(x1 - x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.m = m
        self.dist_func = dist_func
        self._relief = Relief_jl.relief


    def fit(self, data, target):
        """
        Rank features using Relief feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._relief(data, target, self.m, self.dist_func)
        self.rank = rankdata(-self.weights, method='ordinal')

        # Return reference to self
        return self


    def transform(self, data):
        """
        Perform feature selection using computed feature ranks.

        Args:
            data (numpy.ndarray): matrix of data samples on which to perform feature selection.

        Returns:
            (numpy.ndarray): result of performing feature selection.
        """

        # select n_features_to_select best features and return selected features.
        msk = self.rank <= self.n_features_to_select  # Compute mask.
        return data[:, msk]  # Perform feature selection.

