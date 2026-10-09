import firedrake as fd
from firedrake import *
import math
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--SFLT', action='store_true', default=False)
parser.add_argument('--nsteps', type=int, default=400)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 6)))
parser.add_argument('--ref_levels', type=int, nargs='+', default=[2, 3, 4, 5])
parser.add_argument('--williamson', type=int, default=2, help='Testcase id in the checkpoint name')
parser.add_argument('--time_degree', type=int, default=1, help='cPG time degree')
args = parser.parse_args()

suffix = "SFLT" if args.SFLT else "pure"
checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
wtc = args.williamson
tdeg = args.time_degree
nrefs = args.ref_levels  # finest = last

all_err_u = {n: [] for n in nrefs[:-1]}
all_err_D = {n: [] for n in nrefs[:-1]}

for seed in args.seeds:
    meshes, u_raw, D_raw = [], [], []
    ok = True
    for n in nrefs:
        path = f"{checkpoint_dir}/velocity_timestepping_w{wtc}_{n}_{args.nsteps}_{suffix}_tdeg{tdeg}_seed{seed}.h5"
        if not os.path.exists(path):
            print(f"  seed {seed}, ref {n}: missing, skipping seed")
            ok = False
            break
        with fd.CheckpointFile(path, 'r') as f:
            mesh = f.load_mesh("sphere" + str(n))
            meshes.append(mesh)
            u_raw.append(f.load_function(mesh, "velocity"))
            D_raw.append(f.load_function(mesh, "depth"))
    if not ok:
        continue

    # Convert to DG for cross-mesh interpolation
    u_dg, D_dg = [], []
    for i, mesh in enumerate(meshes):
        V = VectorFunctionSpace(mesh, "DG", 4)
        Qs = FunctionSpace(mesh, "DG", 3)
        u = Function(V); u.interpolate(u_raw[i]); u_dg.append(u)
        D = Function(Qs); D.interpolate(D_raw[i]); D_dg.append(D)

    # Interpolate everything onto finest mesh
    finest_mesh = meshes[-1]
    V_fine = VectorFunctionSpace(finest_mesh, "DG", 4)
    Q_fine = FunctionSpace(finest_mesh, "DG", 3)
    u_fine_list, D_fine_list = [], []
    for i in range(len(nrefs)):
        u = Function(V_fine); u.interpolate(u_dg[i]); u_fine_list.append(u)
        D = Function(Q_fine); D.interpolate(D_dg[i]); D_fine_list.append(D)

    for i, n in enumerate(nrefs[:-1]):
        all_err_u[n].append(norm(u_fine_list[i] - u_fine_list[-1]))
        all_err_D[n].append(norm(D_fine_list[i] - D_fine_list[-1]))

R0 = 6371220.0  # sphere radius [m]

print(f"\nSpatial convergence  ({suffix}, nsteps={args.nsteps})")
print(f"Mesh: icosahedral sphere, factor-2 refinement between levels")
print(f"Averaged over {len(args.seeds)} seed(s)")
print("=" * 50)

# Theoretical mesh spacing: h ≈ R * sqrt(4π / (10 * 4^n)) for icosahedral mesh
print(f"\n{'ref':>4}  {'ncells':>8}  {'h (km)':>8}")
for n in nrefs:
    ncells = 10 * 4**n
    h_km = R0 * math.sqrt(4 * math.pi / ncells) / 1000.0
    print(f"  {n:>2}   {ncells:>8d}   {h_km:>7.1f}")

print()
avg_u, avg_D = {}, {}
for n in nrefs[:-1]:
    if all_err_u[n]:
        avg_u[n] = np.mean(all_err_u[n])
        avg_D[n] = np.mean(all_err_D[n])
        ncells = 10 * 4**n
        h_km = R0 * math.sqrt(4 * math.pi / ncells) / 1000.0
        print(f"  ref {n} (h≈{h_km:.0f} km): ||u_err||={avg_u[n]:.6e}  ||D_err||={avg_D[n]:.6e}  n={len(all_err_u[n])}")

print("\nConvergence rates (velocity):")
for i in range(len(nrefs) - 2):
    n1, n2 = nrefs[i], nrefs[i + 1]
    if n1 in avg_u and n2 in avg_u:
        rate = math.log(avg_u[n1] / avg_u[n2]) / math.log(2)
        print(f"  ref {n1} -> {n2}: {rate:.4f}")

print("\nConvergence rates (depth):")
for i in range(len(nrefs) - 2):
    n1, n2 = nrefs[i], nrefs[i + 1]
    if n1 in avg_D and n2 in avg_D:
        rate = math.log(avg_D[n1] / avg_D[n2]) / math.log(2)
        print(f"  ref {n1} -> {n2}: {rate:.4f}")
