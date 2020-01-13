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

test_img = X_train[0:1]
test_label = labels_train[0:1]
# test_list = X_train[0:20]
# for img in test_list:
#   plt.imshow(np.reshape(img, (-1, 28)), cmap="gray")
#   plt.show()
print(test_label[0])

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

def d_sigmoid(x, y):
  return y*(1-y)

def relu(x):
  return np.maximum(0,x)

def d_relu(x, y):
  res = np.ones(y.shape)
  res[y < 0] = 0
  return res

print(sigmoid(4.0))

#%%
class Neurons():
  def __init__(self, active_fn, d_active_fn, init_weight):
    self.active_fn = active_fn
    self.d_active_fn = d_active_fn
    self.output = 0
    self.weight = init_weight

  def set_weight(self, weight):
    self.weight = np.array(weight)

  def get_output(self, input_arr):
    self.input_arr = self.input_arr
    self.output = self.active_fn(input_arr.dot(self.weight))
    return self.output

  def get_gradient(self):
    return self.d_active_fn(self.output) * self.input_arr

#%%
class Layer():
  def __init__(self, neu_num, active_fn_type, input_size, init_weight_type):
    self.neu_list = []
    self.input_size = input_size
    self.input_arr=np.zeros(input_size)
    self.neu_num = neu_num
    self.output_arr = np.zeros(neu_num)
    for i in range(neu_num):
      self.neu_list.append(Neurons(*self.get_active_fn(active_fn_type), self.get_init_weight(input_size, init_weight_type)))

  def get_active_fn(self, fn_type):
    if fn_type == ActiveFnType.sigmoid:
      return sigmoid, d_sigmoid
    if fn_type == ActiveFnType.none:
      return lambda x:x, lambda x, y: np.ones(y.shape)
    if fn_type == ActiveFnType.relu:
      return relu, d_relu

  def get_output(self, input_arr):
    print("you must realize this abstract method")
    return np.array([])
  
  def get_gradient(self):
    print("you must realize this abstract method")
    return np.array([])

  def get_init_weight(self, input_size, init_type):
    size = input_size + 1

    if init_type == NeuInitType.zero:
      return np.zeros(size)
    if init_type == NeuInitType.random:
      return np.random.randn(size) / np.sqrt(size / 2)

  def get_weight(self):
    return np.array(list(map(lambda neu: neu.weight, self.neu_list)))

  def set_weight(self):
    print("you must realize this abstract method")

class FullConnectLayer(Layer):
  def __init__(self, neu_num, active_fn_type, input_size, init_weight_type):
    self.input_size = input_size
    self.weight_matrix = self.get_init_matrix(neu_num, input_size, init_weight_type)
    # self.input_arr=np.zeros(input_size)
    # self.output_arr = np.zeros(neu_num)
    self.neu_num = neu_num
    self.active_fn, self.d_active_fn = self.get_active_fn(active_fn_type)

  def get_init_matrix(self, neu_num, input_size, init_weight_type):
    matrix = []
    for i in range(neu_num):
      matrix.append(self.get_init_weight(input_size, init_weight_type))
    return np.array(matrix)

  def get_output(self, input_arr):
    input_arr = np.array(input_arr)
    if input_arr.ndim == 1:
      input_arr.shape(1, -1)
    row, column = input_arr.shape
    assert(column == self.input_size)

    input_arr = np.column_stack((input_arr, np.ones(row)))
    self.input_arr = input_arr

    self.output_arr = self.acc_get_output(input_arr, self.weight_matrix)
    return self.output_arr

  def acc_get_output(self, input_arr, weight_matrix):
    return self.active_fn(np.dot(input_arr, weight_matrix.T))
  
  def get_gradient(self, pre_g):
    active_fn_g = self.d_active_fn(self.input_arr, self.output_arr)
    gradient = -np.dot((pre_g * active_fn_g).T, self.input_arr)
    chain_d = np.dot((pre_g * active_fn_g), np.delete(self.get_weight(), -1, axis=1))
    return gradient, chain_d

  def set_weight(self, weight_matrix):
    self.weight_matrix = weight_matrix

  def get_weight(self):
    return self.weight_matrix

#%%
def softmax(output_arr):
  epsilon = 1e-8
  max_val = np.max(output_arr)
  exp_arr = np.exp(output_arr-max_val)
  return exp_arr / (np.sum(exp_arr, axis=1).reshape(-1,1) + epsilon)

def d_output(x, y):
  return 1

def cross_entropy(y, label):
  epsilon = 1e-8
  return -np.diag(np.dot(label, np.log(y+epsilon).T))

def d_loss(y, label):
  return y - label

class ClassifyOutputLayer(FullConnectLayer):
  def __init__(self, output_size, input_size, init_weight_type,
   output_fn=softmax,
   d_output_fn=d_output):
    self.output_fn = output_fn
    self.d_output_fn = d_output_fn
    super().__init__(output_size, None, input_size, init_weight_type)

  def get_active_fn(self, active_fn_type):
    return self.output_fn, self.d_output_fn

#%%
def adam(pre_m, pre_v, cur_g, learning_rate,
    alpha,
    belta,
    epsilon
  ):
    m = alpha * pre_m + (1 - alpha) * cur_g
    v = belta * pre_v + (1 - belta) * (np.power(cur_g, 2))
    return m, v, -learning_rate * m / (np.power(v, 0.5) + epsilon)

# def test_adam(loop, x,
#   alpha=0.9,
#   belta=0.99,
#   learning_rate=0.01,
#   epsilon=10 ** -8
# ):
#   pre_m = 0
#   pre_v = 0
#   for i in range(loop):
#     g = d_test_fn(x)
#     pre_m, pre_v, g = adam(pre_m, pre_v, g, learning_rate,
#       alpha,
#       belta,
#       epsilon
#     )
#     x=x+g
#     print(x)
#   return x

# def test_fn(X_train):
#   return 1/2 * np.dot(X_train, X_train.T)

# def d_test_fn(x):
#   return x

# test_arr = np.array([1, 300, 20, 1])
# print(test_fn(test_arr))
# print(test_fn(test_adam(20000, test_arr)))

#%%
def adam_optimizer(network,  input_arr, res_arr, learning_rate,
    alpha=0.9,
    belta=0.99,
    epsilon=10 ** -8
  ):
    loss_derivative = network.get_d_loss(input_arr, res_arr)
    pre_derivative = loss_derivative
    for layer in network.hidden_layer[::-1]:
      gradient, pre_derivative = layer.get_gradient(pre_derivative)
      pre_m = layer.adam_m if hasattr(layer, "adam_m") else np.zeros(gradient.shape)
      pre_v = layer.adam_v if hasattr(layer, "adam_v") else np.zeros(gradient.shape)
      m, v, delta = adam(pre_m, pre_v, gradient, learning_rate,
        alpha,
        belta,
        epsilon
      )
      theta = layer.get_weight()
      layer.set_weight(theta + delta)
      layer.adam_m = m
      layer.adam_v = v

#%%
class LayerType():
  full_connect = 0

class NetworkOutputType():
  classify = 0

class OptimiserType():
  adam = 0

class Network():
  def __init__(self, input_size, output_size, output_type,
      hidden_layer_list, 
      active_fn_type=ActiveFnType.sigmoid, 
      neu_init_type=NeuInitType.random,
      optimizer=adam_optimizer,
    ):
    self.optimizer=optimizer
    self.input_size = input_size
    self.output_size = output_size
    self.output_type = output_type
    self.cur_output_arr = np.zeros(output_size)
    self.output_fn, self.d_output_fn = self.get_output_fn(output_type)
    self.loss_fn, self.d_loss_fn = self.get_loss_fn(output_type)
    self.active_fn_type = active_fn_type
    self.neu_init_type = neu_init_type
    self.hidden_layer = self.get_hidden_layer(hidden_layer_list)

  def get_output_fn(self, output_type):
    if output_type == NetworkOutputType.classify:
     return softmax, d_output

  def get_loss_fn(self, output_type):
    if output_type == NetworkOutputType.classify:
      return cross_entropy, d_loss

  def get_current_input_size(self):
    return self.input_size if len(self.hidden_layer) == 0 else self.hidden_layer[-1].neu_num

  def get_hidden_layer(self, layer_list):
    hidden_layer = []
    input_size = self.input_size

    for (neu_num, layer_type) in layer_list:
      if layer_type == LayerType.full_connect:
        hidden_layer.append(FullConnectLayer(neu_num, self.active_fn_type, input_size, self.neu_init_type))
      input_size = hidden_layer[-1].neu_num

    hidden_layer.append(ClassifyOutputLayer(self.output_size, input_size, self.neu_init_type,
      self.output_fn,
      self.d_output_fn
    ))
    return hidden_layer

  def get_output(self, input_arr):
    res = input_arr
    for layer in self.hidden_layer:
      res = layer.get_output(res)

    self.cur_output_arr = res 
    return np.array(res)

  def get_loss(self, input_arr, res_arr):
    res = self.get_output(input_arr)
    return self.loss_fn(res, res_arr)
  
  def get_d_loss(self, input_arr, res_arr):
    res = self.get_output(input_arr)
    # print(self.loss_fn(res, res_arr))
    return self.d_loss_fn(res, res_arr)

  def fit(self, train_X, train_Y,
      batch_size = 1,
      epoch=1,
      learning_rate = 0.01
    ):
    train_arr = list(zip(train_X, train_Y))
    size = len(train_arr)
    for j in range(epoch):
      for i in range(0, size, batch_size):
        if i + batch_size < size:
          sample, label = zip(*train_arr[i:i+batch_size])
        # else:
          # sample, label = zip(*(np.append(train_arr[i:],train_arr[:batch_size+1-size])))
        self.optimizer(self, np.array(sample), np.array(label), learning_rate)
        print(np.mean(self.get_loss(sample, label)))

#%%
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
bp = Network(X_train.shape[1], 10, NetworkOutputType.classify, bp_layer
# , active_fn_type=ActiveFnType.relu
)
bp.fit(normalize_img(X_train)[:12000], get_num_onehot(labels_train)[:12000], 32
# ,learning_rate=10
)

# print(bp.get_loss(test_img, test_label))

# %%
res = bp.get_output(X_test)
score = 0
for y, label in zip(res, labels_test):
  # print(np.argmax(y), label)
  if np.argmax(y) == label:
    score+=1

print(score, labels_test.shape)

# print(bp.get_output(X_train[3:6]))
# print(np.argmax(bp.get_output(X_test), axis=1))

# %%
