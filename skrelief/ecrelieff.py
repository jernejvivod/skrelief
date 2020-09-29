import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl


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
        If equal to None, the default implementation of L1 distance in Julia is used.
        mode (string): specifies which type of weights update to perform and can either have the value of "k_nearest", "diff" or "exp_rank" (see reference paper). 
        sig (float): kernel width used when mode has the value of "exp_rank".
        f_type (string): specifies whether the features are continuous or discrete 
        and can either have the value of "continuous" or "discrete".

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k (int): number of nearest neighbours to find (for each class).
        dist_func (function): function used to measure similarities between samples.
        mode (string): which type of weights update to perform.
        sig (float): kernel width (when mode has the value of "exp_rank").
        f_type (string): continuous or discrete features.

    Author: Jernej Vivod
    """
  

    def __init__(self, n_features_to_select=10, m=-1, k=5, dist_func=None, mode="k_nearest", sig=1.0, f_type="continuous"):
        self.n_features_to_select = n_features_to_select
        self.m = m                                        
        self.k = k                                        
        self.dist_func = dist_func
        self.mode = mode
        self.sig = sig
        self.f_type = f_type


    def fit(self, data, target):
        """
        Rank features using Evaporative Cooling Relieff feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """
        
        # Rank features.
        if self.dist_func is not None:
            # If distance function specified.
            self.rank = Relief_jl.ecrelieff(data, target, self.m, self.k, self.dist_func, 
                    mode=self.mode, sig=self.sig, f_type=self.f_type)
        else:
            # If distance function not specified, use default L1 distance (implemented in Julia).
            self.rank = Relief_jl.ecrelieff(data, target, self.m, self.k, 
                    mode=self.mode, sig=self.sig, f_type=self.f_type)

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

