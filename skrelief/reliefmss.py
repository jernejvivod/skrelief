import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import ReliefMSS as ReliefMSS_jl

class ReliefMSS(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the ReliefMSS algorithm

    Reference:
        Salim Chikhi and Sadek Benhammada. ReliefMSS: A variation on a
        feature ranking ReliefF algorithm. IJBIDM, 4:375–390, 2009.

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
        _reliefmss (function): function implementing ReliefF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, m=-1, k=5, dist_func=lambda x1, x2 : np.sum(np.abs(x1-x2), 1)):
        self.n_features_to_select = n_features_to_select
        self.m = m 
        self.k = k
        self.dist_func = dist_func
        self._reliefmss = ReliefMSS_jl.reliefmss


    def fit(self, data, target):
        """
        Rank features using ReliefMSS feature selection algorithm

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
            pass
            # warnings.warn("Parameter k was reduced to {0} because one of the classes " \
            #         "does not have {1} instances associated with it.".format(min_instances, self.k), Warning)

        # Compute feature weights and rank.
        self.weights = self._reliefmss(data, target, self.m, int(min(self.k, min_instances)), self.dist_func)
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

