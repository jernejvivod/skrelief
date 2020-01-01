import numpy as np
import os
from sklearn.metrics import pairwise_distances
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import IRelief as IRelief_jl

class IRelief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the IRelief algorithm.
    
    Reference:
        Yijun Sun and Jian Li. Iterative RELIEF for feature weighting. In ICML
        2006 - Proceedings of the 23rd International Conference on Machine
        Learning, volume 2006, pages 913–920, 2006.
    
    Args:
        n_features_to_select (int): number of features to select from dataset.
        max_iter (int): maximum number of iterations to perform. 
        k_width (int): Kernel width. 
        conv_condition (float): threshold for change in feature weights at which to declare convergence.
        initial_w_div (float): initial value with which to divide the feature weights.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        max_iter (int): maximum number of iterations to perform. 
        k_width (int): Kernel width. 
        conv_condition (float): threshold for change in feature weights at which to declare convergence.
        initial_w_div (float): initial value with which to divide the feature weights.
        _irelief (function): function implementing IRelief written in Julia programming language.

    Author: Jernej Vivod
    """
    
    def __init__(self, n_features_to_select=10, max_iter=100,
            k_width=5, conv_condition=1.0e-12, initial_w_div=1):
        self.n_features_to_select = n_features_to_select  # number of features to select
        self.max_iter = max_iter                          # Maximum number of iterations
        self.k_width = k_width                            # kernel width
        self.conv_condition = conv_condition              # convergence condition
        self.initial_w_div = initial_w_div                # initial weight quotient
        self._irelief = IRelief_jl.irelief


    def fit(self, data, target):
        """
        Rank features using I-Relief feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Run I-RELIEF feature selection algorithm.
        self.weights = self._irelief(data, target, self.max_iter, self.k_width, 
                self.conv_condition, self.initial_w_div)
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
