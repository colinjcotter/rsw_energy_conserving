import firedrake as fd
from firedrake import *
import math
import os
import argparse
import numpy as np

# Temporal error vs the exact Läuter et al. (2005) Example 3 solution (--williamson 2).
# Fix the mesh (--ref_level), vary the number of steps, compare each run to the analytic
# solution at the final time.  For --pure the rates are the cPG temporal order; for SFLT
# ||u - u_exact|| is the physical stochastic deviation (no rates).
# Keep the exact-solution block in sync with sw_tools.py testcase==2 and exact_cnv_analysis.py.

parser = argparse.ArgumentParser()
parser.add_argument('--SFLT', action='store_true', default=False)
parser.add_argument('--williamson', type=int, default=2, help='Testcase id in the checkpoint name')
parser.add_argument('--ref_level', type=int, default=5)
parser.add_argument('--timesteps', type=int, nargs='+', default=[25, 50, 100, 200, 400, 800])
parser.add_argument('--tmax', type=float, default=864000.0, help='Final time [s] (sets exact-solution phase)')
parser.add_argument('--time_degree', type=int, default=1, help='cPG time degree')
parser.add_argument('--degree', type=int, default=1, help='DG degree of the depth space Q (matches sw_tools args.degree)')
parser.add_argument('--seed', type=int, default=987654321)
args = parser.parse_args()

suffix = "SFLT" if args.SFLT else "pure"
checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"

# physical constants (must match sw_tools.py)
R0 = 6371220.0
Omega = 7.292e-5
g = 9.8
H = 5960.0
deg = args.degree  # DG degree of the depth space Q (must match the run's args.degree)


def exact_fields(mesh, tval):
    """Läuter Example 3 exact (u, D) at time tval.  Keep in sync with sw_tools testcase==2."""
    x, y, z = SpatialCoordinate(mesh)
    alpha = pi/4.0
    u_0 = 2.0*pi*R0/(12.0*24.0*3600.0)
    k1 = Constant(133681.0)
    k2 = Constant(0.0)
    cx = -sin(alpha)*cos(Omega*tval)
    cy = sin(alpha)*sin(Omega*tval)
    cz = cos(alpha)
    cdotn = (cx*x + cy*y + cz*z)/R0
    u_exact = (u_0/R0)*as_vector([cy*z - cz*y, cz*x - cx*z, cx*y - cy*x])
    D_exact = (-0.5*(u_0*cdotn + Omega*z)**2 + (k1 - k2))/g
    return u_exact, D_exact


n = args.ref_level

# load the (fixed) mesh once from any available run, then build the exact solution on it
mesh = None
for ts in args.timesteps:
    path = (f"{checkpoint_dir}/velocity_timestepping_w{args.williamson}_{n}_{ts}_"
            f"{suffix}_tdeg{args.time_degree}_seed{args.seed}.h5")
    if os.path.exists(path):
        with fd.CheckpointFile(path, 'r') as f:
            mesh = f.load_mesh("sphere" + str(n))
        break
if mesh is None:
    print(f"No checkpoints found for ref {n}, {suffix}, tdeg{args.time_degree}, seed{args.seed}")
    raise SystemExit(1)

x, y, z = SpatialCoordinate(mesh)
b_run = Function(FunctionSpace(mesh, "DG", deg)).interpolate(Omega**2*z**2/(2.0*g))
u_ex_ufl, D_ex_ufl = exact_fields(mesh, args.tmax)
Vd = VectorFunctionSpace(mesh, "DG", 4)
Qd = FunctionSpace(mesh, "DG", 3)
u_exd = Function(Vd).interpolate(u_ex_ufl)
D_exd = Function(Qd).interpolate(D_ex_ufl)
norm_u_ex = norm(u_exd)
norm_D_ex = norm(D_exd)

err_u, err_D, rel_u, rel_D = {}, {}, {}, {}
for ts in args.timesteps:
    path = (f"{checkpoint_dir}/velocity_timestepping_w{args.williamson}_{n}_{ts}_"
            f"{suffix}_tdeg{args.time_degree}_seed{args.seed}.h5")
    if not os.path.exists(path):
        print(f"  nsteps {ts}: missing {path}, skipping")
        continue
    with fd.CheckpointFile(path, 'r') as f:
        m = f.load_mesh("sphere" + str(n))
        u_h = f.load_function(m, "velocity")
        eta_h = f.load_function(m, "depth")   # saved field is D - H + b
    xx, yy, zz = SpatialCoordinate(m)
    b_m = Function(FunctionSpace(m, "DG", deg)).interpolate(Omega**2*zz**2/(2.0*g))
    D_h = Function(eta_h.function_space()).assign(eta_h + Constant(H) - b_m)
    Vdm = VectorFunctionSpace(m, "DG", 4)
    Qdm = FunctionSpace(m, "DG", 3)
    u_hd = Function(Vdm).interpolate(u_h)
    D_hd = Function(Qdm).interpolate(D_h)
    u_ex_m = Function(Vdm).interpolate(exact_fields(m, args.tmax)[0])
    D_ex_m = Function(Qdm).interpolate(exact_fields(m, args.tmax)[1])
    err_u[ts] = norm(u_hd - u_ex_m)
    err_D[ts] = norm(D_hd - D_ex_m)
    rel_u[ts] = err_u[ts]/norm(u_ex_m)
    rel_D[ts] = err_D[ts]/norm(D_ex_m)

label = "stochastic deviation ||u-u_exact||" if args.SFLT else "discretisation error"
print(f"\nTemporal error vs exact (Läuter Ex.3)  ({suffix}, ref_level={n}, "
      f"cPG P{args.time_degree}, tmax={args.tmax:.0f}s)")
print(f"Quantity reported: {label}")
print("=" * 60)

print(f"\n{'nsteps':>7}  {'dt (s)':>10}  {'||u_err||':>12}  {'rel_u':>10}  {'||D_err||':>12}  {'rel_D':>10}")
for ts in args.timesteps:
    if ts in err_u:
        dt = args.tmax/ts
        print(f"  {ts:>5}  {dt:>10.1f}  {err_u[ts]:>12.4e}  {rel_u[ts]:>10.4e}  "
              f"{err_D[ts]:>12.4e}  {rel_D[ts]:>10.4e}")

if not args.SFLT:
    tss = [ts for ts in args.timesteps if ts in err_u]
    print("\nConvergence rates (velocity / depth):")
    for i in range(len(tss) - 1):
        t1, t2 = tss[i], tss[i + 1]
        ru = math.log(err_u[t1]/err_u[t2], 2)/math.log(t2/t1, 2)
        rD = math.log(err_D[t1]/err_D[t2], 2)/math.log(t2/t1, 2)
        print(f"  nsteps {t1} -> {t2}:  u {ru:>6.3f}   D {rD:>6.3f}")
else:
    print("\n(SFLT: ||u-u_exact|| is the physical stochastic spread, not a convergent error "
          "-> no rates printed.)")
