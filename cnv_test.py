from sw_tools import *
import firedrake as fd

checkpoint_dir = "../RSW_checkpoint"
nstep = args.nsteps  # or set nstep manually, e.g., nstep = 1000
t0 = tmax

with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{500}.h5", 'r') as afile:
    mesh = afile.load_mesh("sphere")
    V = fd.FunctionSpace(mesh, "BDFM", 2)  # Replace with correct type/degree
    Q = fd.FunctionSpace(mesh, "DG", 1)    # Replace with correct type/degree

    u_true = fd.Function(V)
    D_true = fd.Function(Q)

    u_check = afile.load_function(mesh, "velocity", idx=t0)
    eta_check = afile.load_function(mesh, "depth", idx=t0)

    u_true.interpolate(u_check)
    D_true.interpolate(eta_check)
    print('norm of u_post:', fd.norm(u_true))
    print('norm of D_post:', fd.norm(D_true))