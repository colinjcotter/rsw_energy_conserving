from poisson_tools import *
from petsc4py import PETSc
from FIAT import ufc_simplex, make_quadrature
import os

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

scheme = GalerkinCollocationScheme(order=stages, quadrature_degree=2*stages)

print(type(eqn))
print(type(scheme))
print(type(t), t)
print(type(dT), dT)
print(type(U))
print("stages =", stages, type(stages))

stepper = GalerkinTimeStepper(eqn, scheme, t, dT, U,
                              solver_parameters=solver_parameters)

quadrature = create_time_quadrature(2*stages)
print("Quadrature points:", quadrature.get_points())
print("Quadrature weights:", quadrature.get_weights())
print("Number of stages:", stages)

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

# Projected psi (rotation removed)
psi_perp = fd.Function(Vcg, name="psi_perp")
# Define f = z
z_mode = fd.Function(Vcg, name="zmode")
z_mode.interpolate(z)

outfile = fd.VTKFile(f"{fname}_{suffix}.pvd")

# Vorticity (scalar) on DG space
vorticity = fd.Function(Q, name="vorticity")
vorticity.interpolate(fd.div(perp(Us[0])))

# Rename fields for ParaView — BEFORE first write
Us[0].rename("velocity")
Us[1].rename("F")
Us[2].rename("D")
eta.rename("eta")

# First write with renamed fields
outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, psi_perp, u_noise, vorticity)

# Before using CheckpointFile
checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

t_checkpoint = tmax
tol = 1e-10

u_checkpoint = fd.Function(V, name="velocity_chk")
D_checkpoint = fd.Function(Q, name="depth_chk")

dcount = 0
itcount = 0.0
energy_errs = []

nstep = args.nsteps
nrefs = args.ref_level

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
    #########################################
    # Normalize z_mode to have unit norm
    den = fd.assemble(z_mode * z_mode * dx)
    z_mode.assign(z_mode / np.sqrt(den))
    # Remove the z-component
    num = fd.assemble(psi_noise * z_mode * dx)
    print("Projection coefficient =", num)
    psi_perp.interpolate(psi_noise - num * z_mode)
    print('aftter projection', fd.norm(psi_perp))
    # Verify
    check = fd.assemble(psi_perp * z_mode * dx)
    print("<psi_perp, z_mode> =", check)
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

        print("dU_3 norm:", fd.norm(dU_3))
        print("psi_noise norm:", fd.norm(psi_noise))

        print("noise u norm:", fd.norm(u_noise))
        print("velocity norm:", fd.norm(Us[0]))
        print("ratio:", fd.norm(u_noise)/fd.norm(Us[0]))

        # compute depth and vorticity
        eta.interpolate(D - H + b)
        vorticity.interpolate(fd.div(perp(Us[0])))
        outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, psi_perp, u_noise, vorticity)

# do the checkpointing
with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_{nrefs}.h5", 'w') as afile:
    print("Saving mesh...")
    afile.save_mesh(mesh)
    print("Saving u_checkpoint and D_checkpoint at t0 =", t0)
    u_checkpoint.interpolate(u)
    D_checkpoint.interpolate(D-H+b)
    afile.save_function(u_checkpoint)
    afile.save_function(D_checkpoint)

print('Total linear iterations', itcount)
print('Average linear iterations', itcount/(args.nsteps))