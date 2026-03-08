from sw_tools import *
import firedrake as fd
import math

checkpoint_dir = "../RSW_checkpoint"

# Timesteps to analyze
timesteps = [1000, 2000, 4000, 8000, 20000]

# Load mesh
with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{timesteps[0]}.h5", 'r') as f:
    mesh = f.load_mesh("sphere")

V = fd.FunctionSpace(mesh, "BDFM", 2)
Q = fd.FunctionSpace(mesh, "DG", 1)

# Load all solutions
solutions = {}
for ts in timesteps:
    with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{ts}.h5", 'r') as f:
        u = f.load_function(mesh, "velocity")
        D = f.load_function(mesh, "depth")
        solutions[ts] = {'u': u, 'D': D}
        print(f'norm u_{ts}:', fd.norm(u))

# Reference solution (finest timestep)
u_true = solutions[timesteps[-1]]['u']
print('norm u_true', fd.norm(u_true))



# Direct comparison between consecutive runs
print("u_1000 vs u_2000:", fd.norm(solutions[1000]['u'] - solutions[2000]['u']))
print("u_2000 vs u_4000:", fd.norm(solutions[2000]['u'] - solutions[4000]['u']))# Check if reference is the outlier
print("u_4000 vs u_8000:", fd.norm(solutions[4000]['u'] - solutions[8000]['u']))# Check if reference is the outlier


# Calculate errors
print('\nErrors:')
errors = {}
for ts in timesteps[:-1]:  # Exclude reference
    error = fd.norm(solutions[ts]['u'] - u_true)
    errors[ts] = error
    print(f'error {ts}:', error)

# Calculate convergence rates
print('\nConvergence rates:')
for i in range(len(timesteps) - 2):
    ts1 = timesteps[i]
    ts2 = timesteps[i + 1]
    rate = math.log(errors[ts1] / errors[ts2], 2)
    print(f'rate {ts1}->{ts2}:', rate)
