import numpy as np
from scipy.stats import rankdata

import os

from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import ECRelieff as ECRelieff_jl


class ECRelieff(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the Evaporative Cooling ReliefF algorithm.
    
    Reference:
        B.A. McKinney, D.M. Reif, B.C. White, J.E. Crowe, Jr., J.H. Moore.
        Evaporative cooling feature selection for genotypic data involving interactions.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k (int): number of nearest neighbours to find (for each class).
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k (int): number of nearest neighbours to find (for each class).
        dist_func (function): function used to measure similarities between samples.
        _ec_relieff (function): function implementing ECReliefF algorithm written in Julia programming language.

    Author: Jernej Vivod
    """
  

    def __init__(self, n_features_to_select=10, m=-1, k=5, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.m = m                                        
        self.k = k                                        
        self.dist_func = dist_func                        
        self._ec_relieff = ECRelieff_jl.ec_relieff


    def fit(self, data, target):
        """
        Rank features using Evaporative Cooling Relieff feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """
        
        self.rank = self._ec_relieff(data, target, self.m, self.k, self.dist_func)
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

