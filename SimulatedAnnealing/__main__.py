#%%
import tsplib95

problem = tsplib95.load_problem('./a280.tsp')
solution = tsplib95.load_solution('./a280.opt.tour')

print(problem.dimension)

#%%
