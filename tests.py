import numpy as np
import scipy.io as sio

# Test imports.
from skrelief import relief
from skrelief import relieff
from skrelief import reliefseq
from skrelief import reliefmss
from skrelief import reliefseq
from skrelief import vlsrelief
from skrelief import turf
from skrelief import surf
from skrelief import surfstar
from skrelief import multisurfstar
from skrelief import multisurf
from skrelief import boostedsurf
from skrelief import swrfstar
from skrelief import iterative_relief
from skrelief import irelief
from skrelief import ec_relieff

### Test default initializations and fiting to test data. ###

rbas = dict()
rbas['relief'] = relief.Relief()
rbas['relieff'] = relieff.Relieff()
rbas['reliefseq'] = reliefseq.ReliefSeq()
rbas['reliefmss'] = reliefmss.ReliefMSS()
rbas['reliefseq'] = reliefseq.ReliefSeq()
rbas['surf'] = surf.SURF()
rbas['surfstar'] = surfstar.SURFStar()
rbas['multisurfstar'] = multisurfstar.MultiSURFStar()
rbas['multisurf'] = multisurf.MultiSURF()
rbas['boostedsurf'] = boostedsurf.BoostedSURF()
rbas['swrfstar'] = swrfstar.SWRFStar()
rbas['iterative_relief'] = iterative_relief.IterativeRelief()
rbas['irelief'] = irelief.IRelief()

data1 = sio.loadmat('./test-data/data1.mat')['data']
target1 = np.ravel(sio.loadmat('./test-data/target1.mat')['target'])

for key in rbas.keys():
    rbas[key].fit(data1, target1)

# Wrappers

rbas['vlsrelief'] = vlsrelief.VLSRelief()
rbas['turf'] = turf.TuRF()

data2 = sio.loadmat('./test-data/data2.mat')['data']
target2 = np.ravel(sio.loadmat('./test-data/target2.mat')['target'])

rbas['vlsrelief'].fit(data2, target2)
rbas['turf'].fit(data2, target2)

# Evaporative Cooling ReliefF

rbas['ec_relieff'] = ec_relieff.ECRelieff()
data_ecr = np.round(np.random.rand(1000, 10)).astype(int)
target_ecr = np.logical_xor(data_ecr[:, 0], data_ecr[:, -1])
rbas['ec_relieff'].fit(data_ecr, target_ecr)

#############################################################

