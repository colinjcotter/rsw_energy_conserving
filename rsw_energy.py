from sw_tools import *

problem = NonlinearVariationalProblem(eqn, U)
solver_parameters = {"ksp_type": "preonly",
                     "pc_type": "lu",
                     "pc_factor_mat_solver_type": "mumps"}

solver = NonlinearVariationalSolver(problem,
                                    solver_parameters = solver_parameters)

dT.assign(0.)
solver.solve()
dT.assign(dt)

for step in ProgressBar("Timestep").iter(range(args.nsteps)):
    solver.solve()

    u0.assign(fd.split(U)[::4][-1])
    F0.assign(fd.split(U)[1::4][-1])
    D0.assign(fd.split(U)[3::4][-1])

