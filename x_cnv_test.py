import firedrake as fd
from firedrake import *
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SFLT", action="store_true", default=False)
args = parser.parse_args()

suffix = "SFLT" if args.SFLT else "pure"

checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
nrefs = [2, 3, 4, 5, 6]

# Load all meshes and functions
meshes = []
u_raw = []
D_raw = []
for n in nrefs:
    with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{n}_{suffix}.h5", 'r') as f:
        mesh = f.load_mesh("sphere" + str(n))
        meshes.append(mesh)
        u_raw.append(f.load_function(mesh, "velocity"))
        D_raw.append(f.load_function(mesh, "depth"))

# Interpolate to DG on each mesh (needed for cross-mesh interpolation)
u_dg = []
D_dg = []
for i, mesh in enumerate(meshes):
    V = VectorFunctionSpace(mesh, "DG", 4)
    Q = FunctionSpace(mesh, "DG", 3)
    u = Function(V)
    u.interpolate(u_raw[i])
    u_dg.append(u)
    D = Function(Q)
    D.interpolate(D_raw[i])
    D_dg.append(D)

# Interpolate all levels onto finest mesh for comparison
finest_mesh = meshes[-1]
V_fine = VectorFunctionSpace(finest_mesh, "DG", 4)
Q_fine = FunctionSpace(finest_mesh, "DG", 3)
u_on_fine = []
D_on_fine = []
for i in range(len(nrefs)):
    u = Function(V_fine)
    u.interpolate(u_dg[i])
    u_on_fine.append(u)
    D = Function(Q_fine)
    D.interpolate(D_dg[i])
    D_on_fine.append(D)

# Compute error norms against ref 5 (finest = reference solution)
print("Velocity error norms:")
err_u = []
for i in range(len(nrefs) - 1):
    err = norm(u_on_fine[i] - u_on_fine[-1])
    err_u.append(err)
    print(f"  ||u_{nrefs[i]} - u_{nrefs[-1]}|| = {err:.6e}")

print("\nDepth error norms:")
err_D = []
for i in range(len(nrefs) - 1):
    err = norm(D_on_fine[i] - D_on_fine[-1])
    err_D.append(err)
    print(f"  ||D_{nrefs[i]} - D_{nrefs[-1]}|| = {err:.6e}")

# Convergence rates
print("\nConvergence rates (velocity):")
for i in range(len(err_u) - 1):
    rate = math.log(err_u[i] / err_u[i+1]) / math.log(2)
    print(f"  ref {nrefs[i]} -> {nrefs[i+1]}: rate = {rate:.2f}")

print("\nConvergence rates (depth):")
for i in range(len(err_D) - 1):
    rate = math.log(err_D[i] / err_D[i+1]) / math.log(2)
    print(f"  ref {nrefs[i]} -> {nrefs[i+1]}: rate = {rate:.2f}")
