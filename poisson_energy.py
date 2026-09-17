from poisson_tools import *
from petsc4py import PETSc
from FIAT import ufc_simplex, make_quadrature
import os


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
    "patch_sub_pc_factor_shift_type": "inblocks"
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

pcg = fd.PCG64(seed=args.seed)

rg = fd.RandomGenerator(pcg)
solver_parameters = sparameters

stages = args.time_degree



scheme = ContinuousPetrovGalerkinScheme(order=stages, quadrature_degree=2*stages)
stepper = GalerkinTimeStepper(eqn, scheme, t, dT, U,
                              solver_parameters=solver_parameters)


# quadrature = create_time_quadrature(2*stages)
# print("Quadrature points:", quadrature.get_points())
# print("Quadrature weights:", quadrature.get_weights())
# print("Number of stages:", stages)

Us = U.subfunctions
stagess = stepper.stages.subfunctions
eta = fd.Function(Q)

t0 = 0.
print("cells:", mesh.num_cells(), "  Vdim:", V.dim(), "  Qdim:", Q.dim())
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
elif testcase == 2:
    print("Läuter et al. (2005) Example 3 (precessing solid-body rotation)")
    fname = "lauter_ex3_output/lauter_ex3"
else:
    fname = "rsw_output"

# Projected psi (rotation removed)
psi_perp = fd.Function(Vcg, name="psi_perp") 
# Define f = z
z_mode = fd.Function(Vcg, name="zmode")
z_mode.interpolate(z)


if not args.no_output:
    pvd_path = f"{fname}_{suffix}.pvd"
    pvd_dir = os.path.dirname(pvd_path)
    if pvd_dir:
        os.makedirs(pvd_dir, exist_ok=True)
    outfile = fd.VTKFile(pvd_path)

# Vorticity (scalar) on DG space
vorticity = fd.Function(Q, name="vorticity")
vorticity.interpolate(fd.div(perp(Us[0])))

# Läuter Example 3: exact solution + stochastic-deviation diagnostics (u - u_exact)
if testcase == 2:
    u_exact = fd.Function(V, name="u_exact")
    D_exact = fd.Function(Q, name="D_exact")
    u_dev = fd.Function(V, name="u_dev")
    vorticity_dev = fd.Function(Q, name="vorticity_dev")
    u_exact.project(u_exact_expr)
    D_exact.interpolate(D_exact_expr)
    u_dev.project(Us[0] - u_exact_expr)
    vorticity_dev.interpolate(fd.div(perp(Us[0] - u_exact_expr)))
    extra_out = (u_exact, D_exact, u_dev, vorticity_dev)
else:
    extra_out = ()

if not args.no_output:
    outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, psi_perp, u_noise, vorticity, *extra_out)

# Before using CheckpointFile
checkpoint_dir = "../RSW_checkpoint/new_RSW_checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

t_checkpoint = tmax

tol = 1e-10

u_checkpoint = fd.Function(V, name="velocity")
D_checkpoint = fd.Function(Q, name="depth")

dcount = 0
itcount = 0.0
energy_errs = []
div_norms = []

# --- Exp-1/2 diagnostics CSV (energy + Casimirs + divergence); off unless --diagnostics_every>0 ---
if args.diagnostics_every > 0:
    vort_ufl = fd.div(perp(u))          # relative vorticity (u = split velocity); f = Coriolis (poisson_tools)
    if mesh.comm.rank == 0:
        diag_csv = open(f"w{testcase}_diagnostics_{suffix}.csv", "w")
        diag_csv.write("step,t_days,H,rel_energy_err,mass,total_vorticity,pot_enstrophy,div_L2\n")

# checkpoint for convergence
nstep = args.nsteps  # or use any variable representing your step count
#nrefs = args.ref_level

# --- Noise checkpoint setup ---
noise_checkpoint_dir = "../RSW_checkpoint/noise_ref5"
ref_fine = args.ref_level_fine
noise_checkpoint_path = os.path.join(noise_checkpoint_dir, f"dU3_ref{ref_fine}_seed{args.seed}.h5")

temporal_cnv = args.save_noise or args.load_noise
if args.SFLT:
    if nrefs == ref_fine and not temporal_cnv:
        # Finest mesh (spatial convergence): save dU_3 at each step
        os.makedirs(noise_checkpoint_dir, exist_ok=True)
        noise_chk = fd.CheckpointFile(noise_checkpoint_path, 'w')
        noise_chk.save_mesh(mesh)
    elif nrefs != ref_fine:
        # Coarser mesh: load dU_3 from finest via cross-mesh interpolation
        noise_chk_fine = fd.CheckpointFile(noise_checkpoint_path, 'r')
        mesh_fine = noise_chk_fine.load_mesh("sphere" + str(ref_fine))

# --- Temporal noise checkpoint setup ---
if args.save_noise:
    os.makedirs(args.noise_dir, exist_ok=True)
    dW_time_chk = fd.CheckpointFile(os.path.join(args.noise_dir, f"dW_nsteps{args.nsteps}_ref{nrefs}.h5"), 'w')
    dW_time_chk.save_mesh(mesh)
    dW_named = fd.Function(W_F, name="dW")
elif args.load_noise:
    fine_nsteps = args.nsteps * args.coarsening
    dW_time_chk = fd.CheckpointFile(os.path.join(args.noise_dir, f"dW_nsteps{fine_nsteps}_ref{nrefs}.h5"), 'r')
    mesh_dW = dW_time_chk.load_mesh("sphere" + str(nrefs))

# --- one-time noise sanity check (static: lambda/h ratio) ---
if args.SFLT:
    # analytic icosahedral cell size (CellSize interpolation unsupported in this Firedrake)
    ncells = 10 * 4**nrefs
    h_avg = R0 * (4.0*np.pi/ncells)**0.5
    h_min = h_max = h_avg
    print(f"[sanity] correlation length lam = {lam:.3e} m")
    print(f"[sanity] h_min/avg/max = {h_min:.3e} / {h_avg:.3e} / {h_max:.3e} m")
    print(f"[sanity] lambda/h (avg) = {lam/h_avg:.2f}  (want >~ 3-4)")

for step in fd.ProgressBar("Timestep").iter(range(args.nsteps)):
    count = 0
    for stage in range(stages):
        for dim in range(3):
            stagess[count].assign(Us[dim])
            count += 1
    if args.SFLT:
        # Generate or load dU_3
        if nrefs == ref_fine:
            if args.save_noise:
                dW.assign(rg.normal(W_F, 0.0, 1.0))
                dW_named.assign(dW)
                dW_time_chk.save_function(dW_named, idx=step)
            elif args.load_noise:
                dW.assign(0)
                for k in range(args.coarsening):
                    fine_idx = step * args.coarsening + k
                    dW_fine_k = dW_time_chk.load_function(mesh_dW, "dW", idx=fine_idx)
                    dW.dat.data[:] += dW_fine_k.dat.data_ro[:]
            else:
                dW.assign(rg.normal(W_F, 0.0, 1.0))
            wsolver1.solve()
            wsolver2.solve()
            # wsolver3.solve()         # 3rd smoothing disabled: 2 smoothings -> rougher, smaller-scale noise
            dU_3.assign(dU_2)          # use the 2-smoothed field as the noise
            if not temporal_cnv:
                noise_chk.save_function(dU_3, idx=step)
        else:
            dU_3_fine = noise_chk_fine.load_function(mesh_fine, "dU_3", idx=step)
            dU_3.interpolate(dU_3_fine)
        print(f"[step {step}] ref{nrefs} dU_3 norm = {fd.norm(dU_3):.6e}")
        # Project into noise velocity field
        psi_noise.project(dU_3)
        noise_proj_solver.solve()
    # advancing stepper
    stepper.advance()
    itcount += stepper.solver.snes.getLinearSolveIterations()  
    denergy = (fd.assemble(energy)-energy0)/energy0
    denergy0 = (fd.assemble(energy))
    print(denergy0)
    energy_errs.append(denergy)
    ddiv = fd.norm(fd.div(u))
    print('div(u) L2 norm', ddiv)
    div_norms.append(ddiv)
    if testcase == 5:
        np.savetxt(f"w5_energy_errors_{suffix}.txt", energy_errs)
        np.savetxt(f"w5_div_norms_{suffix}.txt", div_norms)
    elif testcase == 6:
        np.savetxt(f"w6_energy_errors_{suffix}.txt", energy_errs)
        np.savetxt(f"w6_div_norms_{suffix}.txt", div_norms)
    else:
        np.savetxt(f"rsw_energy_errors_{suffix}.txt", energy_errs)
        np.savetxt(f"rsw_div_norms_{suffix}.txt", div_norms)
    # advance time
    t0 += dt
    t.assign(t0)
    print('time', t0)
    # Exp-1/2 diagnostics: energy (denergy0=H, denergy=rel err), Casimirs, divergence
    if args.diagnostics_every > 0 and (step % args.diagnostics_every == 0 or step == args.nsteps - 1):
        mass_c   = fd.assemble(D*dx)
        tot_vort = fd.assemble((vort_ufl + f)*dx)
        pot_ens  = fd.assemble((vort_ufl + f)**2/(2.0*D)*dx)
        if mesh.comm.rank == 0:
            diag_csv.write(f"{step},{t0/86400.0:.6f},{denergy0:.16e},{denergy:.16e},"
                           f"{mass_c:.16e},{tot_vort:.16e},{pot_ens:.16e},{ddiv:.16e}\n")
            diag_csv.flush()
    dcount += 1
    if dcount % args.ndumps == 0:

        if args.SFLT:
            print("dU_3 norm:", fd.norm(dU_3))
            print("psi_noise norm:", fd.norm(psi_noise))
            print("noise u norm:", fd.norm(u_noise))
            print(f"[sanity] u_noise/u ratio = {fd.norm(u_noise)/fd.norm(Us[0]):.4f}  (want ~ 0.01-0.1)")
        print("velocity norm:", fd.norm(Us[0]))
        print(f"[sanity] rel energy err = {denergy:.4e}  (structure: want ~ solver tol)")


        eta.interpolate(D - H + b)
        if testcase == 2:
            u_exact.project(u_exact_expr)
            D_exact.interpolate(D_exact_expr)
            u_dev.project(Us[0] - u_exact_expr)
            vorticity_dev.interpolate(fd.div(perp(Us[0] - u_exact_expr)))
            rel_dev = fd.norm(Us[0] - u_exact_expr)/fd.norm(u_exact_expr)
            print(f"[lauter] ||u - u_exact|| / ||u_exact|| = {rel_dev:.4e}")
        if not args.no_output:
            vorticity.interpolate(fd.div(perp(Us[0])))
            outfile.write(*(Us[i] for i in range(3)), eta, psi_noise, psi_perp, u_noise, vorticity, *extra_out)

# close noise checkpoint
if args.SFLT:
    if nrefs == ref_fine and not temporal_cnv:
        noise_chk.close()
    elif nrefs != ref_fine:
        noise_chk_fine.close()

# close temporal noise checkpoint
if args.save_noise or args.load_noise:
    dW_time_chk.close()

# close diagnostics CSV
if args.diagnostics_every > 0 and mesh.comm.rank == 0:
    diag_csv.close()

# do the checkpointing
with fd.CheckpointFile(f"{checkpoint_dir}/velocity_timestepping_w{testcase}_{nrefs}_{nstep}_{suffix}_tdeg{args.time_degree}_seed{args.seed}.h5", 'w') as afile:
    print("Saving mesh...")
    afile.save_mesh(mesh)
    #if abs(t0 - t_checkpoint) < tol:
    print("Saving u_checkpoint and D_checkpoint at t0 =", t0)
    u_checkpoint.interpolate(u)
    D_checkpoint.interpolate(D-H+b)
    afile.save_function(u_checkpoint)
    afile.save_function(D_checkpoint)
        

print('Total linear iterations', itcount)

print('Average linear iterations', itcount/(args.nsteps))
