import numpy as np
from scipy.stats import rankdata
import os
from sklearn.base import BaseEstimator, TransformerMixin

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')

from julia import TuRF as TuRF_jl
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
        rba (object): feature weighting algorithm wrapped by the TuRF algorithm.

    Attributes:
        n_features_to_select (int): number of features to select from dataset.
        num_it (int): number of iterations.
        _rba (object): feature weighting algorithm wrapped by the TuRF algorithm.
        _turf (function): function implementing TuRF algorithm written in Julia programming language.
    """
   
    def __init__(self, n_features_to_select=10, num_it=50, rba=Relieff()):
        self.n_features_to_select = n_features_to_select
        self.num_it = num_it
        self._rba = rba
        self._turf = TuRF_jl.turf


    def fit(self, data, target):
        """
        Rank features using TuRF feature selection algorithm

        Args:
            data (numpy.ndarray): matrix of data samples
            target (numpy.ndarray): vector of target values of samples

        Returns:
            (object): reference to self
        """

        def rba_wrap(d, t):
            rba = self._rba.fit(d, t)
            return rba.weights, rba.rank

        # Compute feature weights and rank.
        self.weights, self.rank = self._turf(data, target, self.num_it, rba_wrap)
        
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

