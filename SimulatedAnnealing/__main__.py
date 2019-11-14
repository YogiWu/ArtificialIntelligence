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
import math

def simulated_annealing(solution_array, value_fn, 
        begin_t=100000,
        decline=0.9,
        accept_rate=0.01,
        step_count= 100
    ):
        solution = np.array(solution_array)

        end_t = 0.1
        t = begin_t
        k = 1
        n = solution_array.size
        near_area = n * (n - 1) / 2

        while(t > end_t):
            times = 0.0

            while(times < accept_rate*near_area):
                new_solution = np.array(solution)

                i, j = np.random.randint(0, n, 2)
                
                new_solution[[i, j]] = new_solution[[j, i]]

                value = value_fn(new_solution) - value_fn(solution)
                p = math.exp(-max(value, 0)/(k*t))

                times += accept_rate * near_area / step_count * (1/(p+2) + 0.5)
                # accept according the probability
                if (np.random.rand() < p):
                    solution = new_solution
                    times += 1

            t *= decline
        
        return solution
    
new_tour = simulated_annealing(tour_list, get_all_distance)
print(new_tour)
print(get_all_distance(new_tour))    

# %%
