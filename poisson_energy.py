from poisson_tools import *
from petsc4py import PETSc
from FIAT import ufc_simplex, make_quadrature


#print = PETSc.Sys.Print

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
    "snes_monitor": None,
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "snes_rtol": 1e-11,
    "snes_max_it": 10,
    "snes_converged_reason": None,
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    #"ksp_converged_rate": None,
    "ksp_type": "gmres",
    "ksp_atol": 1.0e-50,
    "ksp_max_it": 30,
    "pc_type": "ksp",
    "ksp_ksp_type": "richardson",
    "ksp_richardson_scale": 0.8,
    "ksp_ksp_max_it": 3,
    "ksp" : patch
}

pcg = fd.PCG64(seed=123456789)

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

suffix = "SFLT" if args.SFLT else "pure"
if testcase == 5:
    print("Williamson 5")
    fname = "w5_rsw_output"
elif testcase == 6:
    print("Williamson 6")
    fname = "w6_rsw_output"
else:
    fname = "rsw_output"


outfile = fd.VTKFile(f"{fname}_{suffix}.pvd")

outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, u_noise)

dcount = 0
itcount = 0.0
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
    wsolver3.solve()
    # Compute noise vector field (ufl expression)
    noise_expr = dU_3
    
    psi_noise.project(noise_expr)
    # solver for u_noise
    noise_proj_solver.solve()
    # advancing stepper
    stepper.advance()
    itcount += stepper.solver.snes.getLinearSolveIterations()  
    
    denergy = (fd.assemble(energy)-energy0)/energy0
    denergy0 = (fd.assemble(energy))
    print(denergy0)
    energy_errs.append(denergy)
    if testcase == 5:
        np.savetxt(f"w5_energy_errors_{suffix}.txt", energy_errs)
    elif testcase == 6:
        np.savetxt(f"w6_energy_errors_{suffix}.txt", energy_errs)
    else:
        np.savetxt(f"rsw_energy_errors_{suffix}.txt", energy_errs)
    # advance time
    t0 += dt
    t.assign(t0)
    print('time', t0)
    
    dcount += 1
    if dcount % args.ndumps == 0:
        eta.interpolate(D - H + b)
        outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, u_noise)

print('Total linear iterations', itcount)

print('Average linear iterations', itcount/(args.nsteps))