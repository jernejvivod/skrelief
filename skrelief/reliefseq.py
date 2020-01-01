import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import ReliefSeq as ReliefSeq_jl

class ReliefSeq(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the ReliefSeq algorithm.

    Reference:
        Brett A. McKinney, Bill C. White, Diane E. Grill, Peter W. Li, Ri-
        chard B. Kennedy, Gregory A. Poland, and Ann L. Oberg. ReliefSeq: a
        gene-wise adaptive-k nearest-neighbor feature selection tool for finding
        gene-gene interactions and main effects in mRNA-Seq gene expression
        data. PloS ONE, 8(12):e81527–e81527, Dec 2013.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k_min (int): lower ReliefF k parameter interval bound.
        k_max (int): upper ReliefF k parameter interval bound. 
        dist_func (function): function used to measure similarities between samples.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k_min (int): lower ReliefF k parameter interval bound.
        k_max (int): upper ReliefF k parameter interval bound. 
        dist_func (function): function used to measure similarities between samples.
        _relieff (function): function implementing ReliefF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, m=-1, k_min=5, k_max=10, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.m = m 
        self.k_min = k_min
        self.k_max = k_max
        self.dist_func = dist_func
        self._reliefseq = ReliefSeq_jl.reliefseq


    def fit(self, data, target):
        """
        Rank features using ReliefSeq feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._reliefseq(data, target, self.m, self.k_min, self.k_max, self.dist_func)
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

