import firedrake as fd
import math
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--SFLT', action='store_true', default=False)
parser.add_argument('--williamson', type=int, default=2, help='Testcase id in the checkpoint name')
parser.add_argument('--ref_level', type=int, default=4)
parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 6)))
parser.add_argument('--timesteps', type=int, nargs='+', default=[400, 800, 1600, 3200, 6400])
parser.add_argument('--tmax', type=float, default=100000, help='Total simulation time in seconds')
parser.add_argument('--time_degree', type=int, default=1, help='cPG time degree')
args = parser.parse_args()

suffix = "SFLT" if args.SFLT else "pure"
checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
wtc = args.williamson
nrefs = args.ref_level
timesteps = args.timesteps  # finest = last

all_errors = {ts: [] for ts in timesteps[:-1]}
all_errors_D = {ts: [] for ts in timesteps[:-1]}

for seed in args.seeds:
    finest = f"{checkpoint_dir}/velocity_timestepping_w{wtc}_{nrefs}_{timesteps[-1]}_{suffix}_tdeg{args.time_degree}_seed{seed}.h5"
    if not os.path.exists(finest):
        print(f"  seed {seed}: missing finest, skipping")
        continue
    # Each nsteps run built its own mesh via a separate parallel run, so two
    # files at the same ref_level are NOT guaranteed to share DOF ordering
    # (parallel partitioning is not deterministic across invocations). Load
    # each file onto its own mesh and cross-mesh interpolate onto the finest
    # run's mesh (same pattern as scnv_analysis.py), rather than trusting
    # raw array alignment across independently-partitioned meshes.
    with fd.CheckpointFile(finest, 'r') as f:
        mesh_fine = f.load_mesh("sphere" + str(nrefs))
        u_true_raw = f.load_function(mesh_fine, "velocity")
        D_true_raw = f.load_function(mesh_fine, "depth")
    V_fine = fd.VectorFunctionSpace(mesh_fine, "DG", 4)
    Q_fine = fd.FunctionSpace(mesh_fine, "DG", 3)
    u_true = fd.Function(V_fine); u_true.interpolate(u_true_raw)
    D_true = fd.Function(Q_fine); D_true.interpolate(D_true_raw)

    for ts in timesteps[:-1]:
        path = f"{checkpoint_dir}/velocity_timestepping_w{wtc}_{nrefs}_{ts}_{suffix}_tdeg{args.time_degree}_seed{seed}.h5"
        if not os.path.exists(path):
            print(f"  seed {seed}, nsteps={ts}: missing, skipping")
            continue
        with fd.CheckpointFile(path, 'r') as f:
            mesh_ts = f.load_mesh("sphere" + str(nrefs))
            u_raw = f.load_function(mesh_ts, "velocity")
            D_raw = f.load_function(mesh_ts, "depth")
        u_dg = fd.Function(fd.VectorFunctionSpace(mesh_ts, "DG", 4)); u_dg.interpolate(u_raw)
        D_dg = fd.Function(fd.FunctionSpace(mesh_ts, "DG", 3)); D_dg.interpolate(D_raw)
        u = fd.Function(V_fine); u.interpolate(u_dg)
        D = fd.Function(Q_fine); D.interpolate(D_dg)
        all_errors[ts].append(fd.norm(u - u_true))
        all_errors_D[ts].append(fd.norm(D - D_true))

print(f"\nTemporal convergence  ({suffix}, ref_level={nrefs}, cPG P{args.time_degree})")
print(f"Time stepping: factor-2 refinement between levels, tmax={args.tmax:.0f} s")
print(f"Averaged over {len(args.seeds)} seed(s)")
print("=" * 50)

print(f"\n{'nsteps':>7}  {'dt (s)':>10}  {'dt (hours)':>10}")
for ts in timesteps:
    dt = args.tmax / ts
    print(f"  {ts:>5}   {dt:>10.1f}   {dt/3600:>10.4f}")

print()
avg_errors = {}
for ts in timesteps[:-1]:
    if all_errors[ts]:
        avg = np.mean(all_errors[ts])
        std = np.std(all_errors[ts])
        avg_errors[ts] = avg
        dt = args.tmax / ts
        print(f"  nsteps={ts:5d} (dt={dt:.0f} s): mean={avg:.6e}  std={std:.6e}  n={len(all_errors[ts])}")

print("\n[velocity] Convergence rates:")
ts_avail = [ts for ts in timesteps[:-1] if ts in avg_errors]
for i in range(len(ts_avail) - 1):
    ts1, ts2 = ts_avail[i], ts_avail[i + 1]
    rate = math.log(avg_errors[ts1] / avg_errors[ts2], 2)
    print(f"  rate {ts1}->{ts2}: {rate:.4f}")

print("\n[depth] errors:")
avg_errors_D = {}
for ts in timesteps[:-1]:
    if all_errors_D[ts]:
        avg = np.mean(all_errors_D[ts])
        avg_errors_D[ts] = avg
        dt = args.tmax / ts
        print(f"  nsteps={ts:5d} (dt={dt:.0f} s): mean={avg:.6e}  n={len(all_errors_D[ts])}")

print("\n[depth] Convergence rates:")
tsD_avail = [ts for ts in timesteps[:-1] if ts in avg_errors_D]
for i in range(len(tsD_avail) - 1):
    ts1, ts2 = tsD_avail[i], tsD_avail[i + 1]
    rate = math.log(avg_errors_D[ts1] / avg_errors_D[ts2], 2)
    print(f"  rate {ts1}->{ts2}: {rate:.4f}")
