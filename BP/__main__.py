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
class NeuInitType():
  zero = 0
  random = 1

class ActiveFnType():
  sigmoid = 0
  relu = 1
  none = 2

#%%
def sigmoid(x):
  return 1/(1+np.exp(-x))

def d_sigmoid(x):
  return x*(1-x)

print(sigmoid(4.0))

#%%
class Neurons():
  def __init__(self, active_fn_type, init_weight):
    self.active_fn_type = active_fn_type
    self.active_fn, self.d_active_fn = self.get_active_fn(active_fn_type)
    self.output = 0
    self.weight = init_weight

  def get_active_fn(self, fn_type):
    if fn_type == ActiveFnType.sigmoid:
      return sigmoid, d_sigmoid
    if fn_type == ActiveFnType.none:
      return lambda x:x, lambda x: 1

  def set_weight(self, weight):
    self.weight = np.array(weight)

  def get_output(self, input_arr):
    self.output = self.active_fn(input_arr.dot(self.weight))
    return self.output

#%%
class FullConnectLayer():
  def __init__(self, neu_num, active_fn_type, input_size, init_weight_type):
    self.neu_list = []
    self.input_size = input_size
    self.input_arr=np.zeros(input_size)
    self.neu_num = neu_num
    self.output_arr = np.zeros(neu_num)
    for i in range(neu_num):
      self.neu_list.append(Neurons(active_fn_type, self.get_init_weight(init_weight_type)))

  def get_init_weight(self, init_type):
    size = self.input_size + 1

    if init_type == NeuInitType.zero:
      return np.zeros(size)
    if init_type == NeuInitType.random:
      return np.random.randn(size) / np.sqrt(size / 2)

  def get_output(self, input_arr):
    input_arr = np.array(input_arr)
    assert(input_arr.size == self.input_size)

    input_arr = np.append(input_arr, 1)
    self.input_arr = input_arr

    res = []
    for neu in self.neu_list:
      res.append(neu.get_output(input_arr))

    self.output_arr = np.array(res)
    return self.output_arr

  def get_weight(self):
    return list(map(lambda neu: neu.weight, self.neu_list))

#%%
class OutputLayer(FullConnectLayer):
  def __init__(self, output_size, input_size, init_weight_type, output_fn, loss_fn, d_output_fn, d_loss_fn):
    self.neu_list = []
    self.output_fn = output_fn
    self.d_output_fn = d_output_fn
    self.lose_fn = loss_fn
    self.d_loss_fn = d_loss_fn
    self.input_size = input_size
    self.neu_num = output_size
    for i in range(output_size):
      self.neu_list.append(Neurons(ActiveFnType.none, self.get_init_weight(init_weight_type)))

  def get_output(self, input_arr):
    return self.output_fn(super().get_output(input_arr))

def softmax(output_arr):
  exp_arr = np.exp(output_arr)
  return exp_arr / sum(exp_arr)

def d_softmax(x):
  return np.dot(x, 1-x)

def cross_entropy(y, label):
  return -np.dot(label, np.log(y))

def d_cross_entropy(y, label):
  return label / y

class ClassifyOutputLayer(OutputLayer):
  def __init__(self, output_size, input_size, init_weight_type,
   output_fn=softmax, 
   loss_fn=cross_entropy, 
   d_output_fn=d_softmax, 
   d_loss_fn=d_cross_entropy):
    super().__init__(output_size, input_size, init_weight_type, output_fn, loss_fn, d_output_fn, d_loss_fn)

#%%
class LayerType():
  full_connect = 0

class NetworkOutputType():
  classify = 0

class Network():
  def __init__(self, input_size, output_size, output_type,
      hidden_layer_list, 
      active_fn_type=ActiveFnType.sigmoid, 
      neu_init_type=NeuInitType.random
    ):
    self.input_size = input_size
    self.output_size = output_size
    self.output_type = output_type
    self.active_fn_type = active_fn_type
    self.neu_init_type = neu_init_type
    self.hidden_layer = self.get_hidden_layer(hidden_layer_list)

  def get_current_input_size(self):
    return self.input_size if len(self.hidden_layer) == 0 else self.hidden_layer[-1].neu_num

  def get_hidden_layer(self, layer_list):
    hidden_layer = []
    input_size = self.input_size

    for (neu_num, layer_type) in layer_list:
      if layer_type == LayerType.full_connect:
        hidden_layer.append(FullConnectLayer(neu_num, self.active_fn_type, input_size, self.neu_init_type))
      input_size = hidden_layer[-1].neu_num

    if self.output_type == NetworkOutputType.classify:
      hidden_layer.append(ClassifyOutputLayer(self.output_size, input_size, self.neu_init_type))
    return hidden_layer

  def get_output(self, input_arr):
    res = input_arr
    for layer in self.hidden_layer:
      res = layer.get_output(res)
    return res

#%%
bp_layer = [
  (64, LayerType.full_connect),
  (80, LayerType.full_connect),
  (64, LayerType.full_connect)
]
bp = Network(X_train.shape[1], 10, NetworkOutputType.classify, bp_layer)
print(bp.get_output(test_img))


#%%
def test_fn(X_train):
  return 1/2 * np.sum(X_train * X_train.T)

print(test_fn(np.array([1, 1, 1])))

# %%
