#from sw_tools import *
import firedrake as fd
from firedrake import *
import math

checkpoint_dir = "../RSW_checkpoint"
nrefs = [2, 3, 4, 5, 6]

# Load all meshes and functions
meshes = []
u_raw = []
D_raw = []
for n in nrefs:
    with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{n}.h5", 'r') as f:
        mesh = f.load_mesh("sphere" + str(n))
        meshes.append(mesh)
        u_raw.append(f.load_function(mesh, "velocity"))
        D_raw.append(f.load_function(mesh, "depth"))

# Interpolate to DG on each mesh
u_dg = []
D_dg = []
print(meshes[0].coordinates.function_space().ufl_element())
for i, mesh in enumerate(meshes):
    V = VectorFunctionSpace(mesh, "DG", 3)
    Q = FunctionSpace(mesh, "DG", 2)
    u = Function(V)
    u.interpolate(u_raw[i])
    u_dg.append(u)
    D = Function(Q)
    D.interpolate(D_raw[i])
    D_dg.append(D)

# Interpolate all to finest mesh
finest_mesh = meshes[-1]
V_fine = VectorFunctionSpace(finest_mesh, "DG", 3)
Q_fine = FunctionSpace(finest_mesh, "DG", 2)
u_on_fine = []
D_on_fine = []
for i in range(len(nrefs)):
    u = Function(V_fine)
    u.interpolate(u_dg[i])
    u_on_fine.append(u)
    D = Function(Q_fine)
    D.interpolate(D_dg[i])
    D_on_fine.append(D)

# Compute error norms against finest mesh (truth)
print("Velocity error norms:")
err_u = []
for i in range(len(nrefs) - 1):
    err = norm(u_on_fine[i] - u_on_fine[-1])
    err_u.append(err)
    print(f"  ||u_{nrefs[i]} - u_{nrefs[-1]}|| = {err}")

print("\nDepth error norms:")
err_D = []
for i in range(len(nrefs) - 1):
    err = norm(D_on_fine[i] - D_on_fine[-1])
    err_D.append(err)
    print(f"  ||D_{nrefs[i]} - D_{nrefs[-1]}|| = {err}")

# Compute convergence rates: err[i] / err[i+1]
print("\nConvergence rates (velocity):")
for i in range(len(err_u) - 1):
    ratio = err_u[i] / err_u[i+1]
    rate = math.log(ratio) / math.log(2)
    print(f"  nref {nrefs[i]} -> {nrefs[i+1]}: ratio = {ratio:.4f}, rate = {rate:.2f}")

print("\nConvergence rates (depth):")
for i in range(len(err_D) - 1):
    ratio = err_D[i] / err_D[i+1]
    rate = math.log(ratio) / math.log(2)
    print(f"  nref {nrefs[i]} -> {nrefs[i+1]}: ratio = {ratio:.4f}, rate = {rate:.2f}")

