#%%
import numpy as np
from Network import Network, LayerType, NetworkOutputType

from mnist import MNIST

mndata = MNIST('./MNIST_data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

#%%

