import firedrake as fd
#get command arguments
from petsc4py import PETSc
from firedrake.__future__ import interpolate
from irksome import Dt, MeshConstant, TimeStepper, GalerkinTimeStepper

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Energy conserving SWE on the sphere.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--tmax', type=float, default=1296000, help='Final time in seconds. Default 1296000 (15 days).')
parser.add_argument('--ndumps', type=int, default=10, help='Timesteps per dump. Default 10.')
parser.add_argument('--nsteps', type=int, default=1000, help='Number of steps, default 1000')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--filename', type=str, default='w5')
parser.add_argument('--time_degree', type=int, default=1, help='Degree of polynomials in time.')
parser.add_argument('--bdfm', action='store_true', help='Use the BDFM space.')
parser.add_argument('--centred', action='store_true', help='If present, use the centred scheme for velocity advection in the curl term, otherwise use the upwind scheme.')
parser.add_argument('--williamson', type=int, default=6, help='Williamson testcase number.')

args = parser.parse_known_args()
args = args[0]

tmax = args.tmax

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
nrefs = args.ref_level
name = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

mesh = fd.IcosahedralSphereMesh(radius=R0,
                                refinement_level=nrefs,
                                degree=deg,
                                distribution_parameters
                                =distribution_parameters)
x = fd.SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)

outward_normals = fd.CellNormal(mesh)

def perp(u):
    return fd.cross(outward_normals, u)

assert args.coords_degree == 1, "Need to fix formulation for higher order cells"

degree = args.degree
if args.bdfm:
    family = "BDFM"
else:
    family = "BDM"

E = fd.FunctionSpace(mesh, "CG", degree+2)
V = fd.FunctionSpace(mesh, family, degree+1)
Q = fd.FunctionSpace(mesh, "DG", degree)

# u, F, gamma, m, v, D
W = V * V * V * V * V * Q

dt = args.tmax/args.nsteps

MC = MeshConstant(mesh)
dT = MC.Constant(dt)
t = MC.Constant(0.)

Omega = MC.Constant(7.292e-5)  # rotation rate
g = MC.Constant(9.8)  # Gravitational constant
b = fd.Function(Q, name="Topography")

u0 = fd.Function(V, name="Velocity")
D0 = fd.Function(Q, name="Depth")
eta0 = fd.Function(Q, name="elevation")
psi0 = fd.Function(E, name="streamfunction")

testcase = args.williamson
x, y, z = fd.SpatialCoordinate(mesh)

if testcase == 5:
    u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
    u_max = fd.Constant(u_0)
    u_expr = fd.as_vector([-u_max*y/R0, u_max*x/R0, 0.0])
    eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(z*z/(R0*R0)))/g
    u0.project(u_expr)
    eta0.project(eta_expr)
    # Topography.
    rl = fd.pi/9.0
    lambda_x = fd.atan2(y/R0, x/R0)
    lambda_c = -fd.pi/2.0
    phi_x = fd.asin(z/R0)
    phi_c = fd.pi/6.0
    minarg = fd.min_value(pow(rl, 2),
                          pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
    bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
    b.interpolate(bexpr)
    u0.assign(un)
    D0.assign(eta0 + H - b)

elif testcase == 6:
    lon = fd.atan2(y, x)
    l = (x**2 + y**2)**0.5
    lat = fd.atan2(z, l)

    # code stolen from Alex Brown
    R = fd.Constant(4)
    K = fd.Constant(7.847e-6) # Frequency parameter, in sec^-1
    w = K
    H0 = fd.Constant(8000.)
    psi = fd.Function(E)
    psiexpr = -R0**2 * w * fd.sin(lat) + \
        R0**2 * K * fd.cos(lat)**R * fd.sin(lat) * fd.cos(R*lon)
    psi.interpolate(psiexpr)
    u_expr = perp(fd.grad(psi))
    u0.project(u_expr)
    # Initilising the depth field
    A = (w / 2) * (2 * Omega + w) * fd.cos(lat)**2 + \
        0.25 * K**2 * fd.cos(lat)**(2 * R) * ((R + 1) * fd.cos(lat)**2 + (2 * R**2 - R - 2) - 2 * R**2 * fd.cos(lat)**(-2))
    B_frac = (2 * (Omega + w) * K) / ((R + 1) * (R + 2))
    B = B_frac * fd.cos(lat)**R * ((R**2 + 2 * R + 2) - (R + 1)**2 * fd.cos(lat)**2)
    C = (1 / 4) * K**2 * fd.cos(lat)**(2 * R) * ((R + 1)*fd.cos(lat)**2 - (R + 2))
    Dexpr = H0 + R0**2 * (A + B*fd.cos(lon*R) + C * fd.cos(2 * R * lon))/g
    D0.interpolate(Dexpr)
else:
    raise NotImplementedError

R = 2*Omega*fd.as_vector([0, 0, z])
u1 = fd.Function(V).assign(u0)
F0 = fd.Function(V).project(u0*D0)
m0 = fd.Function(V).project(D0*(u0+R))
gamma0 = fd.Function(V)
w = fd.TestFunction(V)

inner = fd.inner; div = fd.div; grad = fd.grad
dx = fd.dx

#fd.solve(inner(w,gamma0)*dx - div(w)*(inner(u0, u0)/2 +
#                                      inner(R, u0) - g*(D0+b))*dx == 0,
#         gamma0)

U = fd.Function(W)

# u, F, gamma, m, v, D
u, F, gamma, m, v, D = U.subfunctions
u.assign(u0)
#F.assign(F0)
gamma.assign(gamma0)
m.assign(m0)
v.assign(0.)
D.assign(D0)

X = fd.TestFunction(W)

n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)

dS = fd.dS; sign = fd.sign

# build the equations
def u_op(v, m, Pu):
    eqn = div(v)*inner(m, Pu)*dx
    eqn -= div(Pu)*inner(m, v)*dx

    if args.centred:
        Upwind = 0.5
    else:
        Upwind = 0.5 * (sign(fd.dot(Pu, n)) + 1)
    Upwind = 0.5 * (sign(fd.dot(Pu, n)) + 1)
    eqn -= inner(perp(grad(inner(v, perp(Pu)))), m)*dx
    eqn += inner(both(perp(n)*inner(v, perp(Pu))), both(Upwind*m))*dS
    return eqn

# u, F, gamma, m, v, D
u, F, gamma, m, v, D = fd.split(U)
du, dF, dgamma, dm, dv, dD = fd.TestFunctions(W)

# build the equations
# projection of u
eqn = inner(Dt(v),dv)*dx
eqn -= inner(u, dv)*dx
# m projection of dl/du
eqn += inner(Dt(m - D*(u + R)), dm)*dx
# momentum equation
eqn += inner(Dt(m), du)*dx
eqn += u_op(du, m, Dt(v))
eqn += inner(Dt(gamma),du*D)*dx
# F equation
eqn += inner(Dt(F) - Dt(v)*D, dF)*dx
# gamma equation
eqn += inner(Dt(gamma), dgamma)*dx
eqn -= div(dgamma)*(
    inner(u, u)/2
    + inner(R, u)
    - g*(D+b))*dx
# D equation
eqn += (Dt(D) + div(Dt(F)))*dD*dx
