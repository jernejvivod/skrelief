import numpy as np
import importlib


def test_weights_ranks_transform():

    # Define library, modules and RBAs.
    LIB = 'skrelief'
    MODULES = ['relief', 
               'relieff', 
               'reliefseq', 
               'reliefmss', 
               'surfstar', 
               'surf', 
               'multisurfstar',
               'multisurf',
               'swrfstar',
               'boostedsurf',
               'iterative_relief',
               'irelief',
               'ecrelieff',
               'vlsrelief',
               'turf'
               ]
    RBAS = ['Relief', 
            'Relieff',
            'ReliefSeq',
            'ReliefMSS',
            'SURFStar',
            'SURF',
            'MultiSURFStar',
            'MultiSURF',
            'SWRFStar',
            'BoostedSURF',
            'IterativeRelief',
            'IRelief',
            'ECRelieff',
            'VLSRelief',
            'TuRF'
            ] 
    
    # Set column indices for informative features.
    IDX1 = 1
    IDX2 = 3

    # Define data for testing.
    data_fit = np.random.rand(1000, 10)
    data_transform = np.random.rand(1000, 10)
    target = (data_fit[:, IDX1] > data_fit[:, IDX2]).astype(int)

    # Test computation of weights, ranks and transform
    for idx, (module, rba) in enumerate(zip(MODULES, RBAS)):
        print("Testing {0}".format(module))

        # Import module, get RBA class, initialize and fit.
        rba_module = importlib.import_module('{0}.{1}'.format(LIB, module))
        rba_class = getattr(rba_module, rba)
        rba = rba_class(n_features_to_select=2)
        rba.fit(data_fit, target)

        # Check weights
        if module != 'ecrelieff':
            weights = rba.weights
            assert(np.all(weights[IDX1] > weights[np.logical_and(np.arange(len(weights)) != IDX1, np.arange(len(weights)) != IDX2)]))

        # Check rankings
        assert(set(rba.rank[[IDX1, IDX2]]) == {1, 2})

        # Check transform
        transformed = rba.transform(data_transform)
        assert(transformed.shape, (data_transform.shape[0], 2))



if __name__ == '__main__':
    test_weights_ranks_transform()

