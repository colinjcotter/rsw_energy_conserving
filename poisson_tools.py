from sw_tools import *
import math
from petsc4py import PETSc


print = PETSc.Sys.Print

inner = fd.inner; div = fd.div; grad = fd.grad
dx = fd.dx('everywhere', metadata = {'quadrature_degree': 6,
                                   'representation': 'quadrature'})
dS = fd.dS('everywhere', metadata = {'quadrature_degree': 6,
                                   'representation': 'quadrature'})

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

sign = fd.sign

##   Setup noise term using Matern formula   ##
# caculate mesh width
V0 = fd.FunctionSpace(mesh, "DG", 0)
h_cell = fd.CellSize(mesh)

# h_fun = fd.Function(V0, name="h")
# h_fun.interpolate(h_cell)   

# # Now safe to access .dat
# h_vals = h_fun.dat.data_ro
# h_min, h_avg, h_max = float(h_vals.min()), float(h_vals.mean()), float(h_vals.max())

# print("h_min =", h_min)
# print("h_avg =", h_avg)
# print("h_max =", h_max)



# solver_parameters
sp = {"ksp_type": "cg", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}

Vcg = fd.FunctionSpace(mesh, "CG", 3)  
W_F = fd.FunctionSpace(mesh, "DG", 0)
dW = fd.Function(W_F)
dW_phi = fd.TestFunction(Vcg)
dU = fd.TrialFunction(Vcg)
#kappa_inv_sq = fd.Constant((h/10)**2)
dU_1 = fd.Function(Vcg)
dU_2 = fd.Function(Vcg)
dU_3 = fd.Function(Vcg)
noise_scale = fd.Constant(1e8)


nu  = 2.0
lam = 5.0e5         # meters, e.g. ~5*h_avg
kappa = (8.0*nu)**0.5 / lam
kappa_inv_sq = fd.Constant(1.0/(kappa**2))  # = lam**2/(8*nu)

a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx \
            + dU*dW_phi*dx
L_w1 = dW*dW_phi*dx
w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1)
wsolver1 = fd.LinearVariationalSolver(w_prob1, solver_parameters=sp)

L_w2 = dU_1*dW_phi*dx
w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2)
wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)

L_w3 = dU_2*dW_phi*dx
w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3)
wsolver3 = fd.LinearVariationalSolver(w_prob3, solver_parameters=sp)

# Create a  Function to hold the noise velocity field
psi_noise = fd.Function(Vcg, name="psi_noise")  
# build the equations
# u, G, D
u, G, D = fd.split(U)
du, dG, dD = fd.TestFunctions(W)

F = Dt(G)
ubar = F/D

u_noise = fd.Function(V, name="u_noise")
noise_test = fd.TestFunction(V)

lu_parameters = {
    #'snes_monitor': None,
    #'ksp_monitor': None,
    'snes_rtol': 1e-8,
    'snes_atol': 0,
    'snes_stol': 0,
    'ksp_type': 'gmres',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

noise_proj_eq = inner(u_noise - noise_scale*perp(grad(psi_noise)), noise_test)*dx
noise_proj_prob = fd.NonlinearVariationalProblem(noise_proj_eq, u_noise)
noise_proj_solver = fd.NonlinearVariationalSolver(noise_proj_prob, solver_parameters=lu_parameters)

# build the equations
eqn = inner(du, Dt(u))*dx
if args.centred:
    Upwind = 0.5
else:
    Upwind = 0.5 * (sign(fd.dot(u, n)) + 1)
# advection term
if args.SFLT:
    # Standard advection terms 
    eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
    eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
    # additional SFLT noise terms
    eqn -= inner(perp(grad(inner(du, perp(ubar)))), (1/dT)**0.5*noise_scale*perp(grad(psi_noise)))*dx
    eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*((1/dT)**0.5*noise_scale*perp(grad(psi_noise)))))*dS
else:
    # Standard advection terms
    eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
    eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
f = 2*Omega*z/MC.Constant(R0)  # Coriolis parameter
eqn += inner(du, f*perp(ubar))*dx
eqn -= div(du)*(inner(u,u)/2 + g*(D+b))*dx
# G definition
eqn += inner(F - D*u, dG)*dx
# D transport equation
eqn += dD*(Dt(D) + div(F))*dx
