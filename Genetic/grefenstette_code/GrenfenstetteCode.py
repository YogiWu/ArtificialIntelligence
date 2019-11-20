import random

class GrefenstetteCode:
  code = []

  def __init__(self, arrange_array, coded=False):
    if coded:
      self.code = arrange_array
    else:
      self.code = GrefenstetteCode.encoding(arrange_array)

  def get_res(self):
    return GrefenstetteCode.decoding(self.code)

  def cross_with(self, other_code):
    size = len(self.code)

    index1 = random.randint(0, size-1)
    index2 = random.randint(0, size-1)

    res = GrefenstetteCode.cross(self, other_code, min(index1, index2), max(index1, index2))

    return GrefenstetteCode(res[0], True), GrefenstetteCode(res[1], True)

  def mutation(self):
    size = len(self.code)

    index = random.randint(0, size-1)
    random_val = random.randint(1, size - index)
    
    self.code[index] = random_val
    return self

  @staticmethod
  def cross(code1, code2, start, end):
    seg1 = code1.code[start: end]
    seg2 = code2.code[start:end]

    return (
      code1.code[0:start] + seg2 + code1.code[end:],
      code2.code[0:start] + seg1 + code2.code[end:]
    )

  @staticmethod
  def encoding(arrange_array):
    size = len(arrange_array)

    index_array = [i for i in range(1, size+1)]
    new_array = []

    for item in arrange_array:
      index = index_array.index(item)
      new_array.append(index+1)

      index_array.pop(index)  

    return new_array

  @staticmethod
  def decoding(code):
    size = len(code)

    index_array = [i for i in range(1, size+1)]
    new_array = [] 

    for item in code:
      new_array.append(index_array[item - 1])
      index_array.pop(item - 1)
      
    return new_array
