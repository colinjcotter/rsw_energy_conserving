from sw_tools import *

problem = fd.NonlinearVariationalProblem(eqn, U)
initial_solver_parameters = {"ksp_type": "preonly",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps",
                             "snes_monitor": None,
                             "snes_type": "ksponly"}

sparameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    # "snes_max_it": 1,
    # "snes_convergence_test": "skip",
    #"snes_lag_jacobian": -2,
    #"snes_lag_jacobian_persists": None,
    "ksp_monitor": None,
    "ksp_converged_rate": None,
    # "ksp_view": None,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-3,
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

solver = fd.NonlinearVariationalSolver(problem,
                                       solver_parameters = sparameters)

dT.assign(0.)
solver.solve()
dT.assign(dt)

solver_parameters = {"ksp_type": "preonly",
                     "pc_type": "lu",
                     "pc_factor_mat_solver_type": "mumps",
                     "snes_monitor": None}
solver = fd.NonlinearVariationalSolver(problem,
                                       solver_parameters = solver_parameters)


for step in ProgressBar("Timestep").iter(range(args.nsteps)):
    solver.solve()

    u0.assign(fd.split(U)[::4][-1])
    F0.assign(fd.split(U)[1::4][-1])
    D0.assign(fd.split(U)[3::4][-1])

