import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl


class IterativeRelief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the iterative relief algorithm.
    
    Reference: Bruce Draper, Carol Kaito, and Jose Bins. Iterative Relief. Proceedings
        CVPR, IEEE Computer Society Conference on Computer Vision and
        Pattern Recognition., 6:62 – 62, 2003.
    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        min_incl (int): minimum number of samples from each class to include in hypersphere around each sample.
        max_iter (int): maximum number of iterations to perform.
        dist_func (function): function used to measure similarities between samples.
        If equal to None, the default implementation of weighted L1 distance in Julia is used.
        f_type (string): specifies whether the features are continuous or discrete 
        and can either have the value of "continuous" or "discrete".

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        min_incl (int): minimum number of samples from each class to include in hypersphere around each sample.
        max_iter (int): maximum number of iterations to perform.
        dist_func (function): function used to measure similarities between samples.
        f_type (string): continuous or discrete features.

    Author: Jernej Vivod
    """
   
    def __init__(self, n_features_to_select=10, m=-1, min_incl=3, 
            max_iter=100, dist_func=None, f_type="continuous"):
        self.n_features_to_select = n_features_to_select
        self.m = m
        self.min_incl = min_incl
        self.max_iter = max_iter
        self.dist_func = dist_func
        self.f_type = f_type


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
        if self.dist_func is not None:
            # If distance function specified.
            self.weights = Relief_jl.iterative_relief(data, target, self.m, self.min_incl, self.max_iter, self.dist_func, f_type=self.f_type)
        else:
            # If distance function not specified, use default weighted L1 distance (implemented in Julia).
            self.weights = Relief_jl.iterative_relief(data, target, self.m, self.min_incl, self.max_iter, f_type=self.f_type)
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

