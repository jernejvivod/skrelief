import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
Julia(compiled_modules=False)
from julia import Relief as Relief_jl
from skrelief.relieff import Relieff


class TuRF(BaseEstimator, TransformerMixin):
    """sklearn compatible implementation of the TuRF algorithm.

    Reference:
        Jason H. Moore and Bill C. White. Tuning ReliefF for genome-wide
        genetic analysis. In Elena Marchiori, Jason H. Moore, and Jagath C.
        Rajapakse, editors, Evolutionary Computation,Machine Learning and
        Data Mining in Bioinformatics, pages 166–175. Springer, 2007.

    Args:
        n_features_to_select (int): number of features to select from dataset.
        num_it (int): number of iterations.
        rba (object): feature weighting algorithm wrapped by the VLSRelief algorithm. If equal
        to None, the default ReliefF RBA implemented in Julia is used.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        num_it (int): number of iterations.
        _rba (object): feature weighting algorithm wrapped by the TuRF algorithm.
    """
   
    def __init__(self, n_features_to_select=10, num_it=10, rba=None):
        self.n_features_to_select = n_features_to_select
        self.num_it = num_it
        self._rba = rba


    def fit(self, data, target):
        """
        Rank features using TuRF feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        # Compute feature weights and rank.
        if self._rba is not None:
            self.weights = Relief_jl.turf(data, target, self.num_it, self.rba_wrap)
        else:
            self.weights = Relief_jl.turf(data, target, self.num_it)
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

