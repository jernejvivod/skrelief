import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import IterativeRelief as IterativeRelief_jl

class IterativeRelief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the iterative relief algorithm.
    
    Reference: Bruce Draper, Carol Kaito, and Jose Bins. Iterative Relief. Proceedings
        CVPR, IEEE Computer Society Conference on Computer Vision and
        Pattern Recognition., 6:62 – 62, 2003.
    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        min_incl (int): minimum number of samples from each class to include in hypersphere around each sample.
        dist_func (function): function used to measure similarities between samples.
        max_iter (int): maximum number of iterations to perform.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        min_incl (int): minimum number of samples from each class to include in hypersphere around each sample.
        dist_func (function): function used to measure similarities between samples.
        max_iter (int): maximum number of iterations to perform.
        _iterative_relief (function): function implementing IRelief algorithm written in Julia programming language.

    Author: Jernej Vivod
    """
   
    def __init__(self, n_features_to_select=10, m=-1, min_incl=3, 
            dist_func=lambda x1, x2, w : np.sum(np.abs(w*(x1-x2)), 1), max_iter=100):
        self.n_features_to_select = n_features_to_select
        self.m = m
        self.min_incl = min_incl
        self.dist_func = dist_func
        self.max_iter = max_iter
        self._iterative_relief = IterativeRelief_jl.iterative_relief


    def fit(self, data, target):
        """
        Rank features using Iterative Relief feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._iterative_relief(data, target, self.m, self.min_incl, self.dist_func, self.max_iter)
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

