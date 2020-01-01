import numpy as np
import os

from julia import Julia
jl = Julia(compiled_modules=False)
script_path = os.path.abspath(__file__)
jl.eval('push!(LOAD_PATH, "' + script_path[:script_path.rfind('/')] + '/")')
from julia import MBD as MBD_jl

class MBD:
    '''
    Class encapsulating the Mass-based dissimilarity metric.

    Args:
        n_features_to_select (int): Number of i-tree space partitioning models to use.

    Attributes:
        n_features_to_select (int): Number of i-tree space partitioning models to use.
    '''

    def __init__(self, num_itrees=10):
        self.num_itrees = num_itrees

    def get_dist_func(self, data):
        '''
        Get computed metric function from data.

        Args:
            data (numpy.ndarray): Matrix of training samples.

        Returns:
            self: computed metric function implementing mass-based dissimilarity.
        '''

        return MBD_jl.get_dist_func(data, self.num_itrees)

