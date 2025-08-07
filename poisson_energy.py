from poisson_tools import *
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
    #"snes_converged_reason": None,
    #"snes_monitor": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": 1.0e-8,
    "snes_max_it": 10,
    #"ksp_converged_reason": None,
    #"ksp_monitor": None,
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

lu_parameters = {
    'snes_monitor': None,
    #'ksp_monitor': None,
    'snes_rtol': 1e-8,
    'snes_atol': 0,
    'snes_stol': 0,
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

W_F = fd.FunctionSpace(mesh, "DG", 0)
dW = fd.Function(W_F)

pcg = fd.PCG64(seed=1234)

rg = fd.RandomGenerator(pcg)





solver_parameters = sparameters

stages = args.time_degree

ufc_line = ufc_simplex(1)
quadrature = make_quadrature(ufc_line, 2*stages)

stepper = GalerkinTimeStepper(eqn, stages, t, dT, U,
                              quadrature=quadrature,
                              solver_parameters=solver_parameters,
                              options_prefix="rsw")

Us = U.subfunctions
stagess = stepper.stages.subfunctions
eta = fd.Function(Q)

t0 = 0.
print(f"Dt = {dt}")

u, F, D = fd.split(U)
energy = (D*inner(u,u)/2 + g*D*(D/2+b))*dx
energy0 = fd.assemble(energy)
eta.interpolate(D - H + b)

outfile = fd.VTKFile("poisson.pvd")
outfile.write(*(Us[i] for i in range(3)), eta)

dcount = 0
energy_errs = []
for step in fd.ProgressBar("Timestep").iter(range(args.nsteps)):
    count = 0
    for stage in range(stages):
        for dim in range(3):
            stagess[count].assign(Us[dim])
            count += 1
    # setup noise 
    dW.assign(rg.normal(W_F, 0.0, 1.0))
    wsolver1.solve()
    wsolver2.solve()
    #wsolver3.solve()
    # Compute noise vector field (ufl expression)
    noise_expr = dT**0.5*dU_2
    # Project or assign noise_expr into u_noise_func
    psi_noise.project(noise_expr)  # works if spaces match
    # advancing stepper
    stepper.advance()
    denergy = (fd.assemble(energy)-energy0)/energy0
    energy_errs.append(denergy)
    
    t0 += dt
    t.assign(t0)

    dcount += 1
    if dcount % args.ndumps == 0:
        eta.interpolate(D - H + b)
        outfile.write(*(Us[i] for i in range(3)), eta)

np.savetxt("energy_errors.txt", energy_errs)
