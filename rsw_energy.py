from sw_tools import *
from petsc4py import PETSc
from FIAT import ufc_simplex, make_quadrature

print = PETSc.Sys.Print

patch = {
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

sparameters = {
    "snes_converged_reason": None,
    "snes_monitor": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": 1.0e-10,
    "snes_max_it": 10,
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    #"ksp_converged_rate": None,
    "ksp_type": "gmres",
    "ksp_atol": 1.0e-50,
    "ksp_rtol": 1e-10,
    "ksp_max_it": 30,
    "pc_type": "ksp",
    "ksp_ksp_type": "richardson",
    "ksp_max_it": 3,
    "ksp" : patch
}

solver_parameters = sparameters

ufc_line = ufc_simplex(1)
quadrature = make_quadrature(ufc_line, 2)

stepper = GalerkinTimeStepper(eqn, 1, t, dT, U,
                              quadrature=quadrature,
                              solver_parameters=sparameters)

t0 = 0.
for step in fd.ProgressBar("Timestep").iter(range(args.nsteps)):
    stepper.advance()

    t0 += dt
    t.assign(t0)
    
    print(t)
