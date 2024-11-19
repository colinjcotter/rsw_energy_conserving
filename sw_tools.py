import firedrake as fd
#get command arguments
from petsc4py import PETSc
from firedrake.__future__ import interpolate

import mg
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Energy conserving SWE on the sphere.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.')
parser.add_argument('--tmax', type=float, default=1296000, help='Final time in seconds. Default 1296000 (15 days).')
parser.add_argument('--dumpt', type=float, default=86400, help='Dump time in seconds. Default 86400 (24 hours).')
parser.add_argument('--dt', type=float, default=3600, help='Timestep in seconds. Default 1.')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--filename', type=str, default='w5')
parser.add_argument('--time_degree', type=int, default=1, help='Degree of polynomials in time.')
parser.add_argument('--bdfm', action='store_true', help='Use the BDFM space.')

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

basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=ref_level,
                                    degree=deg,
                                    distribution_parameters = distribution_parameters)
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

# u, F, gamma, D
if args.time_degree == 1:
    W = V*V*V*Q
elif args.time_degree == 2:
    W + V*V*V*Q*V*V*V*Q
else:
    raise NotImplementedError

dt = args.dt
dT.assign(dt)

Omega = fd.Constant(7.292e-5)  # rotation rate
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")

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
    bn = fd.Function(V2)
    bn.interpolate(bexpr)
    Dn = fd.Function(Q).assign(etan + H - b)
    return un, Dn, bn

u0, D0, b = Williamson5InitialConditions()
u1 = Function(V).assign(u0)
D0 = Function(Q).assign(D0)
F0 = Function(V).project(u0*D0)

U = Function(W)
X = TestFunctions(W)

us = [u0] + fd.split(U)[::4]
Ds = [D0] + fd.split(U)[1::4]
Fs = [F0] + fd.split(U)[2::4]
gammas = fd.split(U)[3::4]

if args.time_degree == 1:
    Pk_nodes = [0., 1.]
elif args.time_degree == 2:
    Pk_nodes = [0., 0.5, 1.]
else:
    raise NotImplementedError

def make_lagrange(nodes):
    polys = []
    for i in range(args.time_degree+1):
        roots = []
        for j in range(args.time_degree+1):
            if i==j:
                continue
            roots.append(nodes[j])
        polys.append(np.poly(roots))

Pk_basis = make_lagrange(Pk_nodes)

if args.time_degree == 1:
    Pkm1_nodes = [0.5]
elif args.time_degree == 2:
    Pkm1_nodes = [0., 1.]
else:
    raise NotImplementedError

Pkm1_basis = make_lagrange(Pkm1_nodes)

Pk_basis_d = []
for i in range(args.time_degree):
    Pk_basis_d.append(np.polyder(Pk_basis[i]))
Pkm1_basis_d = []
for i in range(args.time_degree-1):
    Pkm1_basis_d.append(np.polyder(Pkm1_basis[i]))

# need to make these the correct degree
degree = 5 # fixme
quad_points, quad_weights = np.polynomial.legendre.leggauss(degree)

# solution at quadrature points
u_quad = []
D_quad = []
F_quad = []
for q in quad_points:
    uval = None
    Dval = None
    Fval = None
    for j in range(args.time_degree):
        if not uval:
            uval = Pk_basis[j](q)*us[j]
            Dval = Pk_basis[j](q)*Ds[j]
            Fval = Pk_basis[j](q)*Fs[j]
        else:
            uval += Pk_basis[j](q)*us[j]
            Dval += Pk_basis[j](q)*Ds[j]
            Fval += Pk_basis[j](q)*Fs[j]
        u_quad.append(uval)
        D_quad.append(Dval)
        F_quad.append(Fval)
        
# time derivative of solution at quadrature points
dudt_quad = []
dDdt_quad = []
for q in quad_points:
    uval = None
    Dval = None
    for j in range(args.time_degree):
        if not uval:
            uval = Pk_basis[j](q)*us[j]
            Dval = Pk_basis[j](q)*Ds[j]
        else:
            uval += Pk_basis_d[j](q)*us[j]
            Dval += Pk_basis_d[j](q)*Ds[j]
        dudt_quad.append(uval)
        dDdt_quad.append(Dval)

# test functions at quadrature points
wus = fd.split(X)[::4]
wFs = fd.split(X)[1::4]
wgammas = fd.split(X)[2::4]
phis = fd.split(X)[3::4]
wu_quad = []
wF_quad = []
wgamma_quad = []
phi_quad = []
for q in quad_points:
    wuval = None
    wFval = None
    wgammaval = None
    phival = None
    for j in range(args.time_degree):
        if not uval:
            wuval = Pkm1_basis[j](q)*wus[j]
            wFval = Pkm1_basis[j](q)*wFs[j]
            wgammaval = Pkm1_basis[j](q)*wgammas[j]
            phival = Pkm1_basis[j](q)*phis[j]
        else:
            wuval += Pkm1_basis[j](q)*wus[j]
            wFval += Pkm1_basis[j](q)*wFs[j]
            wgammaval += Pkm1_basis[j](q)*wgammas[j]
            phival += Pkm1_basis[j](q)*phis[j]
        wu_quad.append(wuval)
        wF_quad.append(wFval)
        wgamma_quad.append(wgammaval)
        phi_quad.append(phival)

# time projection operators
A = np.zeros((args.time_degree-1, args.time_degree-1))
B = np.zeros((args.time_degree-1, args.time_degree))

for qi, q in enumerate(quad_points):
    weight = quad_weights[qi]
    for i in range(args.time_degree-1):
        for j in range(args.time_degree-1):
            A[i,j] += weight*Pkm1_basis[i](q)*Pkm1_basis[j](q)
        for j in range(args.time_degree):
            B[i,j] += weight*Pkm1_basis[i](q)*Pk_basis[j](q)

# projection operator
Proj = np.linalg.solve(A, B)
            
# time projection of u and D at Pkm1 nodes
Pu = []
PD = []
for i in range(args.time_degree-1):
    Pu.append(None)
    PD.append(None)
    for j in range(args.time_degree):
        if not Pu[i]:
            Pu[i] = Proj[i,j]*us[j]
            PD[i] = Proj[i,j]*Ds[j]
        else:
            Pu[i] += Proj[i,j]*us[j]
            PD[i] += Proj[i,j]*Ds[j]

# time projection of u and D at quadrature points
# (plus gamma at quad points)
Pu_quad = []
PD_quad = []
gamma_quad = []
for q in quad_points:
    uval = None
    Dval = None
    gval = None
    for j in range(args.time_degree):
        if not uval:
            uval = Pkm1_basis[j](q)*Pu[j]
            Dval = Pkm1_basis[j](q)*PD[j]
            gval = Pkm1_basis[j](q)*gammas[j]
        else:
            uval += Pkm1_basis[j](q)*Pu[j]
            Dval += Pkm1_basis[j](q)*PD[j]
            gval += Pkm1_basis[j](q)*gammas[j]
        Pu_quad.append(uval)
        PD_quad.append(Dval)
        gamma_quad.append(gval)

dx = fd.dx
n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)

dS = fd.dS

R = f = 2*Omega*as_vector([0, 0, x[2]])

# build the equations
def u_op(v, u, D, gamma):
    F = D*(u+R)
    eqn = - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), F)*dx
    eqn -= fd.inner(both(perp(n)*fd.inner(v, perp(u))), avg(F))*dS
    eqn -= fd.div(v)*inner(F, u)*dx
    eqn += fd.div(u)*inner(F, v)*dx
    eqn -= gamma*v*dx
    return eqn

def F_op(v, u, D, F):
    return fd.inner(F - D*(u + R), v)*dx

def gamma_op(v, u, D, gamma):
    eqn = fd.div(v)*(inner(u, u)/2 + inner(R, u) - g*D)*dx
    eqn -= inner(gamma, v)*dx
    return eqn

def D_op(phi, F):
    return fd.div(F)*phi*dx

# build the time integral
eqn = None
for qi, q in enumerate(quad_points):
    weight = quad_weights[qi]
    # u equation
    if not eqn:
        eqn = weight*fd.inner(D_quad*dudt_quad, wu_quad(qi))*dx
    eqn += weight*fd.inner(dDdt_quad*(u_quad + R), wu_quad(qi))*dx
    eqn += weight*u_op(wu_quad(qi), Pu_quad(qi),
                       D_quad(qi), gamma_quad(qi))
    # F equation
    eqn += weight*F_op(wF_quad(qi), Pu_quad(qi),
                       D_quad(qi), F_quad(qi))
    # gamma equation
    eqn += weight*gamma_op(wgamma_quad(qi), u_quad(qi),
                           D_quad(gi), gamma_quad(qi))
    # D equation
    eqn += weight*(dDdt_quad(qi) + fd.div(F_quad(qi)))*phi_quad(qi)

