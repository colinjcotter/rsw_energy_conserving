from sw_tools import *
import math
from petsc4py import PETSc


print = PETSc.Sys.Print

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

# caculate mesh width for kappa_inv_sq
total_area = fd.assemble(1 * dx(mesh))
num_cells = mesh.num_cells()
avg_area = total_area / num_cells
h = math.sqrt(4 * avg_area / math.sqrt(3))
kappa_inv_sq = fd.Constant((h/2)**2)

# solver_parameters
sp = {"ksp_type": "cg", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}
# Setup noise term using Matern formula
Vcg = fd.FunctionSpace(mesh, "CG", 1) 
W_F = fd.FunctionSpace(mesh, "DG", 0)

dW_phi = fd.TestFunction(Vcg)
dU = fd.TrialFunction(Vcg)

dW = fd.Function(W_F)
dU_1 = fd.Function(Vcg)
dU_2 = fd.Function(Vcg)
dU_3 = fd.Function(Vcg)


a_dW = kappa_inv_sq*fd.inner(fd.grad(dU), fd.grad(dW_phi))*dx \
            + dU*dW_phi*dx
L_w1 = dW*dW_phi*dx
w_prob1 = fd.LinearVariationalProblem(a_dW, L_w1, dU_1, constant_jacobian=True)
wsolver1 = fd.LinearVariationalSolver(w_prob1, solver_parameters=sp)
L_w2 = dU_1*dW_phi*dx
w_prob2 = fd.LinearVariationalProblem(a_dW, L_w2, dU_2, constant_jacobian=True)
wsolver2 = fd.LinearVariationalSolver(w_prob2, solver_parameters=sp)
L_w3 = dU_2*dW_phi*dx
w_prob3 = fd.LinearVariationalProblem(a_dW, L_w3, dU_3,  constant_jacobian=True)
wsolver3 = fd.LinearVariationalSolver(w_prob3, solver_parameters=sp)

# Function to hold the noise velocity field
psi_noise = fd.Function(Vcg, name="u_noise")  
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

# eqn -= inner(perp(grad(inner(du, perp(ubar)))), u)*dx
# eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*u))*dS
# SFLT noise
eqn -= inner(perp(grad(inner(du, perp(ubar)))), u + perp(grad(psi_noise)))*dx
eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*(u+ perp(grad(psi_noise)))))*dS
f = 2*Omega*z/MC.Constant(R0)  # Coriolis parameter
eqn += inner(du, f*perp(ubar))*dx
eqn -= div(du)*(inner(u,u)/2 + g*(D+b))*dx

# build the equations
# G definition
eqn += inner(F - D*u, dG)*dx
# D transport equation
eqn += dD*(Dt(D) + div(F))*dx
