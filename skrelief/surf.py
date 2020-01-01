import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import SURF as SURF_jl

class SURF(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the SURF algorithm.

    Reference:
        Casey S. Greene, Nadia M. Penrod, Jeff Kiralis, and Jason H. Moore.
        Spatially uniform ReliefF (SURF) for computationally-efficient filtering
        of gene-gene interactions. BioData mining, 2(1):5–5, Sep 2009.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        dist_func (function): function used to measure similarities between samples.
        _surf (function): function implementing SURF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.dist_func = dist_func
        self._surf = SURF_jl.surf


    def fit(self, data, target):
        """
        Rank features using SURF feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._surf(data, target, self.dist_func)
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

