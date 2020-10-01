import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl
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
        partition_size (int): size of selected partitions. If equal to -1, use default value computed from data.
        num_partitions_to_select (int): number of partitions to select for each iteration. If equal to
        -1, use default value computed from data.
        num_subsets (int): number of subsets to evaluate.
        rba (object): feature weighting algorithm wrapped by the VLSRelief algorithm. If equal
        to None, the default ReliefF RBA implemented in Julia is used.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        partition_size (int): size of selected partitions.
        num_partitions_to_select (int): number of partitions to select for each iteration.
        num_subsets (int): number of subsets to evaluate.
        rba (object): feature weighting algorithm wrapped by the VLSRelief algorithm.
    """
   
    def __init__(self, n_features_to_select=10, partition_size=-1, num_partitions_to_select=-1, num_subsets=20, rba=None):
        self.n_features_to_select = n_features_to_select
        self.partition_size = partition_size
        self.num_partitions_to_select = num_partitions_to_select
        self.num_subsets = num_subsets
        self.rba = rba


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
        if self.rba is not None:
            # If wrapped RBA specified.
            self.weights = Relief_jl.vlsrelief(data, target, self.partition_size, 
                    self.num_partitions_to_select, self.num_subsets, rba=self.rba)
        else:
            # If wrapped RBA not specified, use default RBA (ReliefF implemented in Julia).
            self.weights = Relief_jl.vlsrelief(data, target, self.partition_size, 
                    self.num_partitions_to_select, self.num_subsets)
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

