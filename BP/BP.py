import numpy as np
from Network import Network, LayerType, NetworkOutputType

from mnist import MNIST

mndata = MNIST('./MNIST_data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

def normalize_img(img_list):
  res = []
  for img in img_list:
    res.append(img / 256)

  return np.array(res)

def get_num_onehot(num_list):
  size = 10
  res = []
  for num in num_list:
    res.append(np.eye(size)[num])
  return np.array(res)

bp_layer = [
  (256, LayerType.full_connect),
  (128, LayerType.full_connect),
  (64, LayerType.full_connect)
]
bp = Network(X_train.shape[1], 10, NetworkOutputType.classify, bp_layer)
bp.fit(normalize_img(X_train), get_num_onehot(labels_train), 32)
bp.save("./BP.pickle")

res = bp.get_output(normalize_img(X_test))
score = 0
for y, label in zip(res, labels_test):
  print(np.argmax(y), label)
  if np.argmax(y) == label:
    score+=1

print(score, labels_test.shape)
