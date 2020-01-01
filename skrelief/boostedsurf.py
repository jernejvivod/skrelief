import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import BoostedSURF as BoostedSURF_jl

class BoostedSURF(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the BoostedSURF algorithm.

    Reference:
        Gediminas Bertasius, Delaney Granizo-MacKenzie, Ryan J. Urba-
        nowicz, and Jason H. Moore. Boosted spatially uniform ReliefF al-
        gorithm for genome-wide genetic analysis. Hanover, NH 03755, USA,
        2013. Dartmouth College.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        phi (int): the phi parameter that controls frequency of distace function weights updates.
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        phi (int): the phi parameter that controls frequency of distace function weights updates.
        dist_func (function): function used to measure similarities between samples.
        _multisurfstar (function): function implementing MultiSURFStar algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, phi=3, dist_func=lambda x1, x2, w : np.sum(w*np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.phi=3
        self.dist_func = dist_func
        self._boostedsurf = BoostedSURF_jl.boostedsurf


    def fit(self, data, target):
        """
        Rank features using BoostedSURF feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._boostedsurf(data, target, self.phi, self.dist_func)
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


