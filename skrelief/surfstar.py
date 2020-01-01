import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import SURFStar as SURFStar_jl

class SURFStar(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the SURFStar algorithm.

    Reference:
        Casey S. Greene, Daniel S. Himmelstein, Jeff Kiralis, and Jason H. Mo-
        ore. The informative extremes: Using both nearest and farthest indivi-
        duals can improve Relief algorithms in the domain of human genetics.
        In Clara Pizzuti, Marylyn D. Ritchie, and Mario Giacobini, editors,
        Evolutionary Computation, Machine Learning and Data Mining in Bi-
        oinformatics, pages 182–193. Springer, 2010.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.
        _surfstar (function): function implementing SURF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.dist_func = dist_func
        self._surfstar = SURFStar_jl.surfstar


    def fit(self, data, target):
        """
        Rank features using SURFStar feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._surfstar(data, target, self.dist_func)
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

