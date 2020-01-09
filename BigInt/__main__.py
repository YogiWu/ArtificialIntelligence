#%%
import numpy as np

a = "75213432434"
b = "03214432406"

# %%
def big_int_to_array(big_int):
  co_array = []

  for i in big_int:
    co_array.append(int(i))

  temp = np.floor(np.log2(len(big_int)))
  zero_arr = np.zeros(int(np.power(2, temp+1)-len(big_int)))
  return np.append(zero_arr, co_array)

print(big_int_to_array(a))

def array_to_big_int(arr):
  res = ""
  index = 0
  for num in arr:
    if int(num) == 0:
      index+=1
    else:
      break

  arr = arr[:index-1:-1]
  for i in range(arr.size):
    if len(res) <= i:
      res += str(int(arr[i]))[::-1]
    else:
      pre_res = res[:i]
      cur_res = res[i:]

      res = pre_res + str(int(cur_res[::-1]) + int(arr[i]))[::-1]

  return res[::-1]

print(array_to_big_int(big_int_to_array(a)))

#%%
def fast_multi(a, b):
  arr_a = big_int_to_array(a)
  arr_b = big_int_to_array(b)
  f_a = np.fft.fft(np.append(np.zeros(len(arr_a)), arr_a))
  f_b = np.fft.fft(np.append(np.zeros(len(arr_b)), arr_b))

  f_c = f_a * f_b
  c = np.fft.ifft(f_c)

  return array_to_big_int(np.round(np.real(c))[:-1])

print(fast_multi(a, b))
print(int(a)*int(b))

#%%
def random_big_int(size):
  res = ''

  for i in range(size):
    res += str(np.random.randint(10))

  return res

size = 1000
a = random_big_int(size)
b = random_big_int(size)
print(a)
print(b)

#%%
import time

start = time.time()
print(fast_multi(a,b))
end = time.time()
print(end-start)
print(int(a)*int(b))

#%%
