import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SFLT", action="store_true", default=False)
args, _ = parser.parse_known_args()

sys.argv = [a for a in sys.argv if a != "--SFLT"]

import firedrake as fd
from firedrake import *
import math

checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
nrefs = [2, 3, 4, 5]

suffix = "SFLT" if args.SFLT else "pure"

meshes = []
u_raw = []
D_raw = []
for n in nrefs:
    fname = f"{checkpoint_dir}/velocity_timestepping_{n}_SFLT.h5" if args.SFLT else f"{checkpoint_dir}/velocity_timestepping_{n}.h5"
    with fd.CheckpointFile(fname, 'r') as f:
        mesh = f.load_mesh("sphere" + str(n))
        meshes.append(mesh)
        u_raw.append(f.load_function(mesh, "velocity_chk"))
        D_raw.append(f.load_function(mesh, "depth_chk"))

# Project to CG on each mesh before cross-mesh interpolation
u_cg = []
D_cg = []
for i, mesh in enumerate(meshes):
    V_cg = VectorFunctionSpace(mesh, "CG", 3)
    Q_cg = FunctionSpace(mesh, "CG", 2)
    u_cg.append(project(u_raw[i], V_cg))
    D_cg.append(project(D_raw[i], Q_cg))

# Cross-mesh interpolate all to finest mesh
finest_mesh = meshes[-1]
V_fine = VectorFunctionSpace(finest_mesh, "CG", 3)
Q_fine = FunctionSpace(finest_mesh, "CG", 2)
u_on_fine = []
D_on_fine = []
for i in range(len(nrefs)):
    u = Function(V_fine)
    u.interpolate(u_cg[i])
    u_on_fine.append(u)
    D = Function(Q_fine)
    D.interpolate(D_cg[i])
    D_on_fine.append(D)

print(f"\n=== {suffix} ===")
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