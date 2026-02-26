import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.insert(0, "src/")
from environment import GridWorldEnvironment
from MDPsolver import MDPsolver
from utils import *
from plot import *
gamma=0.7
gridworld = GridWorldEnvironment(2, 10, prop=0, gamma=gamma)
solver = MDPsolver(gridworld)
policy = solver.soft_value_iteration()

plot_value_and_policy(solver, policy, "softoptimal", "max_ent")

#p_in = np.zeros(100)
#p_in[-1] = 1
occupancy_measure = solver.mu_policy(policy, stochastic=True) #, emp_p_in=p_in)

print(occupancy_measure)
print(occupancy_measure.shape)

plot_on_grid(occupancy_measure, 10, r"OccupancyMeasure $\gamma=$"+str(gamma))

### Comparison with uniform policy
solver = MDPsolver(gridworld)
solver.unif_value_iteration()
plot_value_and_policy(solver, gridworld.uniform_policy(), "uniform", "max_ent")

occupancy_measure = solver.mu_policy(gridworld.uniform_policy(), stochastic=True) #, emp_p_in=p_in)

plot_on_grid(occupancy_measure, 10, r"Uniform for $\gamma=$"+str(gamma))














    

    


