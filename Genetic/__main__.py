#%%
import tsplib95

# problem = tsplib95.load_problem('./a280.tsp')
# solution = tsplib95.load_solution('./a280.opt.tour')

problem = tsplib95.load_problem('./ch150.tsp')
solution = tsplib95.load_solution('./ch150.opt.tour')

#%%
import numpy as np

dim = problem.dimension

tour_list = np.array(range(1, dim+1))

#%%
def get_instance(point_1, point_2):
    return problem.wfunc(point_1, point_2)

def get_all_distance(point_array):
    dis = 0

    for i in range(point_array.size):
        dis += get_instance(point_array[i], point_array[(i+1) % (point_array.size-1)])

    return dis

print(tour_list)
print(get_all_distance(tour_list))

print(solution.tours[0])
print(get_all_distance(np.array(solution.tours[0])))

#%%
from grefenstette_code.GrenfenstetteCode import GrefenstetteCode as Code

#%%
import numpy as np
import random
from functools import cmp_to_key

def genetic(generate_list, value_fn,
        generate_list_count=50,
        generate_count=200,
        mutate_rate=0.01
    ):
        for i in range(generate_count):
            value_list = list(map(value_fn, generate_list))
            min_val = min(value_list)
            pro_list = np.cumsum(np.array(value_list)-min_val)

            if pro_list[-1] == 0.0:
                break

            pro_list = pro_list / pro_list[-1]
            sub_list = []
            for j in range(0, generate_list_count, 2):
                index1 = np.where(pro_list > random.random())[0][0]
                code1 = generate_list[index1]

                index2 = np.where(pro_list > random.random())[0][0]
                code2 = generate_list[index2]

                sub_list.extend(code1.cross_with(code2))

            for j in range(int(mutate_rate*len(sub_list))):
                sub_list[random.randint(0, len(sub_list)-1)].mutation()

            generate_list = sub_list

        generate_list.sort(key=cmp_to_key(lambda pre, cur: value_fn(pre) - value_fn(cur)))
        value_list = list(map(value_fn, generate_list))
        return generate_list[-1]

#%%
# generate_count = 50
# generate_list = [Code(random.sample(list(tour_list), len(tour_list))) for i in range(generate_count)]

# solution = genetic(generate_list, lambda item:1/get_all_distance(np.array(item.get_res())), generate_count=generate_count)
# print(solution.get_res())
# print(get_all_distance(np.array(solution.get_res())))

#%%
class Code:
    code = 0.0

    def __init__(self, code):
        self.code = code

    def get_res(self):
        return self.code
    
    def cross_with(self, other_code):
        alpha = random.random()

        res = Code.cross(self, other_code, alpha)

        return Code(res[0]), Code(res[1])

    def mutation(self):
        alpha = random.uniform(-1,1) / 100

        self.code += alpha
        return self

    @staticmethod
    def cross(code1, code2, alpha):
        liner_mix = lambda num1, num2: num1*alpha+num2*(1-alpha)

        return liner_mix(code1.code, code2.code),liner_mix(code2.code, code1.code)

#%%
import math
generate_count = 50
generate_list = np.linspace(0, 1, generate_count)

def fn(x):
    return x*math.sin(5*x)

solution = genetic(list(map(lambda x: Code(x), generate_list)), lambda x: fn(x.get_res()), generate_count=generate_count)
print(solution.get_res())
print(fn(solution.get_res()))

#%%