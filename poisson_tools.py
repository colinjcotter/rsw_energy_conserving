from sw_tools import *

inner = fd.inner; div = fd.div; grad = fd.grad
dx = fd.dx

# F = G_t
# u, G, D
W = V * V * Q

U = fd.Function(W)

u, G, D = U.subfunctions
u.assign(u0)
D.assign(D0)

n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)

dS = fd.dS; sign = fd.sign

# build the equations
# u, G, D
u, G, D = fd.split(U)
du, dG, dD = fd.TestFunctions(W)

F = Dt(G)
ubar = F/D

eqn = inner(du, Dt(u))*dx
if args.centred:
    Upwind = 0.5
else:
    Upwind = 0.5 * (sign(fd.dot(u, n)) + 1)
eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
f = 2*Omega*z/MC.Constant(R0)  # Coriolis parameter
eqn += inner(du, f*perp(ubar))*dx
eqn -= div(du)*(inner(u,u)/2 + g*(D+b))*dx

# build the equations
# G definition
eqn += inner(F - D*u, dG)*dx
# D transport equation
eqn += dD*(Dt(D) + div(F))*dx

q = fd.TrialFunction(E)
p = fd.TestFunction(E)

un, _, _ = fd.split(U)

qn = fd.Function(E, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'preonly',
           'pc_type':'lu',
           "pc_factor_mat_solver_type": "superlu_dist"}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)
