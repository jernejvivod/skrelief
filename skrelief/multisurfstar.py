import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import MultiSURFStar as MultiSURFStar_jl

class MultiSURFStar(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the MultiSURFStar algorithm.

    Reference:
        Delaney Granizo-Mackenzie and Jason H. Moore. Multiple threshold
        spatially uniform ReliefF for the genetic analysis of complex human
        diseases. In Leonardo Vanneschi, William S. Bush, and Mario Giacobini,
        editors, Evolutionary Computation, Machine Learning and Data Mining
        in Bioinformatics, pages 1–10. Springer, 2013.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.
        _multisurfstar (function): function implementing MultiSURFStar algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.dist_func = dist_func
        self._multisurfstar = MultiSURFStar_jl.multisurfstar


    def fit(self, data, target):
        """
        Rank features using MultiSURFStar feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._multisurfstar(data, target, self.dist_func)
        self.rank = rankdata(-self.weights, method='ordinal')
        
        # Return reference to self.
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


