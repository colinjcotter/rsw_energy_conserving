import firedrake as fd
from firedrake import *
import math
import os
import argparse
import numpy as np

# Error vs the exact Läuter et al. (2005) Example 3 solution (testcase --williamson 2).
# Unlike scnv/tcnv (run-vs-reference), this compares each run against the analytic
# solution at the final time.  For --pure this is the true discretisation error and
# the reported rates are meaningful; for SFLT it is the physical stochastic deviation
# (||u - u_exact||) which does NOT converge under refinement.

parser = argparse.ArgumentParser()
parser.add_argument('--SFLT', action='store_true', default=False)
parser.add_argument('--williamson', type=int, default=2, help='Testcase id in the checkpoint name')
parser.add_argument('--nsteps', type=int, default=1000)
parser.add_argument('--ref_levels', type=int, nargs='+', default=[3, 4, 5])
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


err_u, err_D, rel_u, rel_D = {}, {}, {}, {}

for n in args.ref_levels:
    path = (f"{checkpoint_dir}/velocity_timestepping_w{args.williamson}_{n}_{args.nsteps}_"
            f"{suffix}_tdeg{args.time_degree}_seed{args.seed}.h5")
    if not os.path.exists(path):
        print(f"  ref {n}: missing {path}, skipping")
        continue
    with fd.CheckpointFile(path, 'r') as f:
        mesh = f.load_mesh("sphere" + str(n))
        u_h = f.load_function(mesh, "velocity")
        eta_h = f.load_function(mesh, "depth")   # saved field is D - H + b

    x, y, z = SpatialCoordinate(mesh)
    # rebuild the same DG-degree topography that the run subtracted, so D_h = eta_h + H - b
    b_run = Function(FunctionSpace(mesh, "DG", deg)).interpolate(Omega**2*z**2/(2.0*g))
    D_h = Function(eta_h.function_space()).assign(eta_h + Constant(H) - b_run)

    u_ex_ufl, D_ex_ufl = exact_fields(mesh, args.tmax)

    # high-order DG for an accurate L2 error (mirrors scnv_analysis)
    Vd = VectorFunctionSpace(mesh, "DG", 4)
    Qd = FunctionSpace(mesh, "DG", 3)
    u_hd = Function(Vd).interpolate(u_h)
    u_exd = Function(Vd).interpolate(u_ex_ufl)
    D_hd = Function(Qd).interpolate(D_h)
    D_exd = Function(Qd).interpolate(D_ex_ufl)

    err_u[n] = norm(u_hd - u_exd)
    err_D[n] = norm(D_hd - D_exd)
    rel_u[n] = err_u[n]/norm(u_exd)
    rel_D[n] = err_D[n]/norm(D_exd)

label = "stochastic deviation ||u-u_exact||" if args.SFLT else "discretisation error"
print(f"\nError vs exact (Läuter Ex.3)  ({suffix}, nsteps={args.nsteps}, "
      f"cPG P{args.time_degree}, tmax={args.tmax:.0f}s)")
print(f"Quantity reported: {label}")
print("=" * 60)

print(f"\n{'ref':>4}  {'h (km)':>8}  {'||u_err||':>12}  {'rel_u':>10}  {'||D_err||':>12}  {'rel_D':>10}")
for n in args.ref_levels:
    if n in err_u:
        ncells = 10 * 4**n
        h_km = R0 * math.sqrt(4 * math.pi / ncells) / 1000.0
        print(f"  {n:>2}  {h_km:>8.1f}  {err_u[n]:>12.4e}  {rel_u[n]:>10.4e}  "
              f"{err_D[n]:>12.4e}  {rel_D[n]:>10.4e}")

if not args.SFLT:
    refs = [n for n in args.ref_levels if n in err_u]
    print("\nConvergence rates (velocity / depth):")
    for i in range(len(refs) - 1):
        n1, n2 = refs[i], refs[i + 1]
        ru = math.log(err_u[n1]/err_u[n2], 2)
        rD = math.log(err_D[n1]/err_D[n2], 2)
        print(f"  ref {n1} -> {n2}:  u {ru:>6.3f}   D {rD:>6.3f}")
else:
    print("\n(SFLT: ||u-u_exact|| is the physical stochastic spread, not a convergent error "
          "-> no rates printed.)")
