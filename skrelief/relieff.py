import numpy as np
from scipy.stats import rankdata
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl


class Relieff(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the ReliefF algorithm.

    Reference:
        Marko Robnik-Šikonja and Igor Kononenko. Theoretical and empirical
        analysis of ReliefF and RReliefF. Machine Learning, 53(1):23–69, Oct
        2003.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k (int): number of nearest neighbours to find (for each class).
        dist_func (function): function used to measure similarities between samples.
        If equal to None, the default implementation of L1 distance in Julia is used.
        mode (str): how to compute updated weights from nearest neighbours. 
        Legal values are "k_nearest", "diff" and "exp_rank"
        sig (float): parameter specifying the influence of weights. Ignored if mode = "k_nearest".
        f_type (string): specifies whether the features are continuous or discrete 
        and can either have the value of "continuous" or "discrete".

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        m (int): training data sample size.
        k (int): number of nearest neighbours to find (for each class).
        dist_func (function): function used to measure similarities between samples.
        mode (str): how to compute updated weights from nearest neighbours. 
        sig (float): parameter specifying the influence of weights. Ignored if mode = "k_nearest".
        f_type (string): continuous or discrete features.
        _relieff (function): function implementing ReliefF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, m=-1, k=10, dist_func=None, 
            mode="k_nearest", sig=1.0, f_type="continuous"):
        self.n_features_to_select = n_features_to_select
        self.m = m 
        self.k = k
        self.dist_func = dist_func
        self.mode = mode
        self.sig = sig
        self.f_type = f_type


    def fit(self, data, target):
        """
        Rank features using Relieff feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Get number of instances with class that has minimum number of instances.
        _, instances_by_class = np.unique(target, return_counts=True)
        min_instances = np.min(instances_by_class) - 1
       
        # If class with minimal number of examples (minus one) has less than k examples, issue warning
        # that parameter k was reduced.
        if min_instances < self.k:
            warnings.warn("Parameter k was reduced to {0} because one of the classes " \
                    "does not have {1} instances associated with it.".format(min_instances, self.k), Warning)

        # Compute feature weights and rank.
        if self.dist_func is not None:
            # If distance function specified.
            self.weights = Relief_jl.relieff(data, target, self.m, int(min(self.k, min_instances)),self.dist_func, mode=self.mode, sig=self.sig, f_type=self.f_type)
        else:
            # If distance function not specified, use default L1 distance (implemented in Julia).
            self.weights = Relief_jl.relieff(data, target, self.m, int(min(self.k, min_instances)), mode=self.mode, sig=self.sig, f_type=self.f_type)
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

