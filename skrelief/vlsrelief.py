import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import VLSRelief as VLSRelief_jl
from skrelief.relieff import Relieff

class VLSRelief(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the VLSRelief algorithm.

    Reference:
        Margaret Eppstein and Paul Haake. Very large scale ReliefF for genome-
        wide association analysis. In 2008 IEEE Symposium on Computational
        Intelligence in Bioinformatics and Computational Biology, CIBCB ’08,
        2008.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        num_partitions_to_select (int): number of partitions to select for each iteration.
        num_subsets (int): number of subsets to evaluate.
        partition_size (int): size of selected partitions.
        rba (object): feature weighting algorithm wrapped by the VLSRelief algorithm.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        num_partitions_to_select (int): number of partitions to select for each iteration.
        num_subsets (int): number of subsets to evaluate.
        partition_size (int): size of selected partitions.
        _rba (object): feature weighting algorithm wrapped by the VLSRelief algorithm.
        _vlsrelief (function): function implementing VLSRelief algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, num_partitions_to_select=10, num_subsets=50, partition_size=10, rba=Relieff()):
        self.n_features_to_select = n_features_to_select
        self.num_partitions_to_select = num_partitions_to_select
        self.num_subsets = num_subsets
        self.partition_size = partition_size
        self._rba = rba
        self._vlsrelief = VLSRelief_jl.vlsrelief


    def fit(self, data, target):
        """
        Rank features using VLSRelief feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        self.weights = self._vlsrelief(data, target, self.num_partitions_to_select, 
                self.num_subsets, self.partition_size, lambda d, t : self._rba.fit(d, t).weights)
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

