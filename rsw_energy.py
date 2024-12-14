from sw_tools import *
from petsc4py import PETSc

print = PETSc.Sys.Print
problem = fd.NonlinearVariationalProblem(eqn, U)

sparameters = {
    "snes_converged_reason": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_atol": 1.0e-8,
    "snes_max_it": 10,
    "ksp_converged_reason": None,
    "ksp_converged_rate": None,
    "ksp_type": "gmres",
    "ksp_atol": 1.0e-50,
    "ksp_rtol": 1e-6,
    "ksp_max_it": 30,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": True,
    "patch_pc_patch_sub_mat_type": "seqdense",
    "patch_pc_patch_construct_dim": 0,
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_local_type": "additive",
    "patch_pc_patch_precompute_element_tensors": True,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
    "patch_sub_pc_factor_shift_type": "nonzero"
}

solver_parameters = sparameters

solver = fd.NonlinearVariationalSolver(problem,
                                       solver_parameters = solver_parameters)

# initial guess
Us = U.subfunctions
for i in range(args.time_degree-1):
    Us[4*i].assign(u0)
    Us[4*i+1].assign(F0)
    Us[4*i+2].assign(D0)

dT.assign(dt)
print(f"dt = {dt}")

for step in fd.ProgressBar("Timestep").iter(range(args.nsteps)):
    for i in range(args.time_degree-1):
        Us[4*i].assign(u0)
        Us[4*i+1].assign(F0)
        Us[4*i+2].assign(D0)
    
    solver.solve()

    u0.assign(Us[::4][-1])
    F0.assign(Us[1::4][-1])
    D0.assign(Us[3::4][-1])
