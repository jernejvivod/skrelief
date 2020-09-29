import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl


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
        mode (str): how to compute updated weights from nearest neighbours. 
        Legal values are "k_nearest", "diff" and "exp_rank"
        sig (float): parameter specifying the influence of weights. Ignored if mode = "k_nearest".
        f_type (string): specifies whether the features are continuous or discrete 
        and can either have the value of "continuous" or "discrete".

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k_min (int): lower ReliefF k parameter interval bound.
        k_max (int): upper ReliefF k parameter interval bound. 
        dist_func (function): function used to measure similarities between samples.
        mode (str): how to compute updated weights from nearest neighbours. 
        sig (float): parameter specifying the influence of weights. Ignored if mode = "k_nearest".
        f_type (string): continuous or discrete features.
    """
   
    def __init__(self, n_features_to_select=10, m=-1, k_min=5, k_max=10, 
            dist_func=None, mode="k_nearest", sig=1.0, f_type="continuous"):
        self.n_features_to_select = n_features_to_select
        self.m = m 
        self.k_min = k_min
        self.k_max = k_max
        self.dist_func = dist_func
        self.mode = mode
        self.sig = sig
        self.f_type = f_type


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
        if self.dist_func is not None:
            self.weights = Relief_jl.reliefseq(data, target, self.m, self.k_min, self.k_max, 
                    self.dist_func, mode=self.mode, sig=self.sig, f_type=self.f_type)
        else:
            self.weights = Relief_jl.reliefseq(data, target, self.m, self.k_min, self.k_max, 
                    mode=self.mode, sig=self.sig, f_type=self.f_type)
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

