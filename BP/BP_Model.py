#%%
import pickle

with open('./BP/BP.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
  bp = pickle.load(f)
  f.close()

#%%
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

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

from mnist import MNIST
def main():
  mndata = MNIST('./MNIST_data/')
  # X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())

  res = bp.get_output(normalize_img(X_test))
  score = 0
  for y, label in zip(res, labels_test):
    print(np.argmax(y), label)
    if np.argmax(y) == label:
      score+=1

  print(score, labels_test.shape)

  # with open('./BP/BP.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
  #   bp = pickle.load(f)
  #   f.close()
  # path = "./BP/test" #文件夹目录
  # files= os.listdir(path) #得到文件夹下的所有文件名称
  # s = []
  # for file in files: #遍历文件夹
  #   img = Image.open(path+"/"+file).convert('L')
  #   plt.imshow(img, cmap="gray")
  #   plt.show()
  #   print(np.argmax(bp.get_output(normalize_img(np.array(img).reshape(1, -1)))))

# %%
