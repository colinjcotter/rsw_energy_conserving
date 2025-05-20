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
parser.add_argument('--dumpt', type=float, default=86400, help='Dump time in seconds. Default 86400 (24 hours).')
parser.add_argument('--nsteps', type=int, default=1000, help='Number of steps, default 1000')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--filename', type=str, default='w5')
parser.add_argument('--time_degree', type=int, default=1, help='Degree of polynomials in time.')
parser.add_argument('--bdfm', action='store_true', help='Use the BDFM space.')
parser.add_argument('--centred', action='store_true', help='If present, use the centred scheme for velocity advection in the curl term, otherwise use the upwind scheme.')

args = parser.parse_known_args()
args = args[0]

tmax = args.tmax
dumpt = args.dumpt

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

degree = args.degree
if args.bdfm:
    family = "BDFM"
else:
    family = "BDM"

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

def Williamson5InitialConditions():
    x = fd.SpatialCoordinate(mesh)
    u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
    u_max = fd.Constant(u_0)
    u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
    eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
    un = fd.Function(V, name="Velocity").project(u_expr)
    etan = fd.Function(Q, name="Elevation").project(eta_expr)
    
    # Topography.
    rl = fd.pi/9.0
    lambda_x = fd.atan2(x[1]/R0, x[0]/R0)
    lambda_c = -fd.pi/2.0
    phi_x = fd.asin(x[2]/R0)
    phi_c = fd.pi/6.0
    minarg = fd.min_value(pow(rl, 2),
                          pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
    bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
    bn = fd.Function(Q)
    bn.interpolate(bexpr)
    Dn = fd.Function(Q).assign(etan + H - b)
    return un, Dn, bn

u0, D0, b = Williamson5InitialConditions()
R = 2*Omega*fd.as_vector([0, 0, x[2]])
u1 = fd.Function(V).assign(u0)
D0 = fd.Function(Q).assign(D0)
F0 = fd.Function(V).project(u0*D0)
m0 = fd.Function(V).project(D0*(u0+R))
gamma0 = fd.Function(V)
w = fd.TestFunction(V)

inner = fd.inner; div = fd.div
dx = fd.dx

fd.solve(inner(w,gamma0)*dx - div(w)*(inner(u0, u0)/2 +
                                      inner(R, u0) - g*(D0+b))*dx == 0,
         gamma0)

U = fd.Function(W)

# u, F, gamma, m, v, D
u, F, gamma, m, v, D = U.subfunctions
u.assign(u0)
F.assign(F0)
gamma.assign(gamma0)
m.assign(m0)
v.assign(0.)
D.assign(D0)

X = fd.TestFunction(W)

n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)

dS = fd.dS

# build the equations
def u_op(v, m, u, Pu, D, gamma):
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    eqn = - fd.inner(perp(fd.grad(fd.inner(v, perp(Pu)))), m)*dx
    eqn -= fd.inner(both(perp(n)*fd.inner(v, perp(Pu))), both(Upwind*m))*dS
    eqn += fd.div(v)*fd.inner(m, Pu)*dx
    eqn -= fd.div(Pu)*fd.inner(m, v)*dx
    return eqn

def F_op(v, u, D, F):
    return fd.inner(F - D*u, v)*dx

def D_op(phi, F):
    return fd.div(F)*phi*dx

# u, F, gamma, m, v, D
u, F, gamma, m, v, D = fd.split(U)
du, dF, dgamma, dm, dv, dD = fd.TestFunctions(W)

# build the equations
# projection of u
eqn = inner(Dt(v),dv)*dx
eqn -=  inner(u, dv)*dx
# m projection of dl/du
eqn += inner(Dt(m),dm)*dx
eqn -= inner(Dt(D*(u + R)), dm)*dx
# momentum equation
eqn += inner(Dt(m), du)*dx
eqn += u_op(du, m, u, Dt(v), D, gamma)
eqn += fd.inner(gamma,du)*dx
# F equation
eqn += inner(Dt(F), dF)*dx
eqn -= inner(Dt(u*D), dF)*dx
# gamma equation
eqn += inner(Dt(gamma), dgamma)*dx
eqn -= div(dgamma)*Dt(inner(u, u)/2 + inner(R, u) - g*(D+b))*dx
# D equation
eqn += Dt(D)*dD*dx
eqn -= D_op(dD, F)
