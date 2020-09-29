# skrelief


This package contains implementations of interfaces for various Relief-based algorithms
implemented in the Relief.jl package for the Julia programming lanugage. The implementations
are accessed using the PyJulia interface. The interfaces are interoperable with scikit-learn 
machine learning workflows.


## Installation

The package [Relief.jl](https://github.com/jernejvivod/Relief.jl) has to be installed to access the implementations of the algorithms.
You can install it manually or use the Julia's package manager.

You should be able to install skrelief with a pip command:

```
pip install -e git+https://github.com/Architecton/skrelief#egg=skrelief
```

## Usage

The implementations are compatible with the scikit-learn framework interface and provide
the *fit*, *transform* and *fit_transfrom* methods for evaluating features and performing
feature selection.

For example, the SWRF* algorithm can be used to estimate feature importances as follows:

```python

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from skrelief import swrfstar

# Load sample dataset.
dataset = sio.loadmat('./sample-data/ionosphere.mat')
data = dataset['data']
target = np.ravel(dataset['target'])

# Initialize feature standardizer.
scaler = StandardScaler()

# Initialize SWRFStar implementation instance.
fs = swrfstar.SWRFStar(n_features_to_select=10)

# Fit to standardized data.
fs.fit(scaler.fit_transform(data), target)

print(fs.weights)
>>> [-0.06888013  0.         -0.00866773 -0.00068636  0.01457658  0.01989347
0.02516213  0.03677635  0.03489531  0.03712191  0.04198217  0.02944513
0.07086742  0.02104629  0.07813643  0.02211294  0.0639499   0.00891899
0.06179809  0.0123553   0.06572316  0.01179598  0.0488017   0.02123056
0.03963257  0.00414455  0.02850292  0.0022764   0.0214597   0.00125613
0.01835287  0.00151015  0.02770807  0.01127832]

print(fs.rank)
>>> [34 31 33 32 22 20 15 10 11  9  7 12  2 19  1 16  4 26  5 23  3 24  6 18
8 27 13 28 17 30 21 29 14 25]

```

The implementations can be integrated directly into scikit-learn workflows. An example of using the ReliefF
algorithm in a scikit-learn pipeline is shown below:


```python

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from skrelief import swrfstar

# Load sample dataset.
dataset = sio.loadmat('./sample-data/ionosphere.mat')
data = dataset['data']
target = np.ravel(dataset['target'])

# Initialize SWRFStar implementation instance.
fs = swrfstar.SWRFStar(n_features_to_select=10)

# Initialize classification pipeline.
clf = Pipeline(steps=[
                      ('scaler', StandardScaler()), 
                      ('fs', swrfstar.SWRFStar(n_features_to_select=10)), 
                      ('rf', RandomForestClassifier(n_estimators=100))
                     ])

# Split data into training and test sets.
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=1)

# Fit to training data.
clf.fit(data_train, target_train)

# Perform classification on test data.
score = clf.score(data_test, target_test)
print(score)
>>> 0.8591549295774648

```

