#%%
import numpy as np

#%%
from mnist import MNIST

mndata = MNIST('./MNIST_data/')
X_train, labels_train = map(np.array, mndata.load_training())
X_test, labels_test = map(np.array, mndata.load_testing())

print(X_train.shape, labels_train.shape)

# %%
import matplotlib.pyplot as plt

test_img = X_train[0]
plt.imshow(np.reshape(test_img, (-1, 28)), cmap="gray")
plt.show()
print(labels_train[0])

#%%
class Neurons():
  def __init__(self, active_fn, weight):
    self.active_fn = active_fn
    self.output = 0
    self.set_weight(weight)

  def set_weight(self, weight):
    self.weight = np.array(weight)

  def get_output(self, input_arr):
    input_arr = np.array(input_arr)
    assert(input_arr.size + 1 == self.weight.size)

    self.output = self.active_fn(np.sum(np.append(input_arr, 1) * self.weight))
    return self.output

class NeuInitType():
  zero = 0
  random = 1

#%%
def sigmoid(x):
  return 1/(1+np.exp(x))

print(sigmoid(4.0))

#%%
class FullConnectLayer():
  def __init__(self, neu_num, active_fn, init_weight):
    self.neu_list = []
    self.neu_num = neu_num
    for i in range(neu_num):
      self.neu_list.append(Neurons(active_fn, init_weight))

  def get_output(self, input_arr):
    res = []
    for neu in self.neu_list:
      res.append(neu.get_output(input_arr))
    return np.array(res)

  def get_weight(self):
    return list(map(lambda neu: neu.weight, self.neu_list))

class LayerType():
  full_connect = 0

class Network():
  def __init__(self, input_size, output_size, output_fn,
      hidden_layer_list, 
      active_fn=sigmoid, 
      neu_init_type=NeuInitType.zero
    ):
    self.input_size = input_size
    self.output_size = output_size
    self.output_fn = output_fn
    self.active_fn = active_fn
    self.neu_init_type = neu_init_type
    self.hidden_layer = self.get_hidden_layer(hidden_layer_list)

  def get_init_weight(self, size):
    if self.neu_init_type == NeuInitType.zero:
      return np.zeros(size)
    if self.neu_init_type == NeuInitType.random:
      return np.random.rand(size)

  def get_current_input_size(self):
    return self.input_size if len(self.hidden_layer) == 0 else self.hidden_layer[-1].neu_num

  def get_hidden_layer(self, layer_list):
    hidden_layer = []
    input_size = self.input_size

    for (neu_num, layer_type) in layer_list:
      if layer_type == LayerType.full_connect:
        hidden_layer.append(FullConnectLayer(neu_num, self.active_fn, self.get_init_weight(input_size + 1)))
      input_size = hidden_layer[-1].neu_num

    hidden_layer.append(FullConnectLayer(self.output_size, self.active_fn, self.get_init_weight(input_size + 1)))
    return hidden_layer

  def get_output(self, input_arr):
    res = input_arr
    for layer in self.hidden_layer:
      res = layer.get_output(res)
    return self.output_fn(res)

#%%
def softmax(output_arr):
  exp_arr = np.exp(output_arr)
  return exp_arr / sum(exp_arr)

bp_layer = [
  (64, LayerType.full_connect),
  (80, LayerType.full_connect),
  (64, LayerType.full_connect)
]
bp = Network(X_train.shape[1], 10, softmax, bp_layer)
print(bp.get_output(test_img))

for layer in bp.hidden_layer:
  print(layer.get_weight())

#%%
