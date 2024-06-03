# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import underworld3 as uw
import sympy
from underworld3.function import expression


# %%
import math
rho = expression(r'\uprho', 10)
three = expression(r'\textrm{[iii]}', 3)
kappa = expression(r'\upkappa', 100)
alpha = expression(r'\alpha', 3e-5)
pi = expression(r'\pi', math.pi )


# %%
three

# %% [markdown]
# Now let's build a mesh and some mesh variables
#
#

# %%
mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), 
        maxCoords=(1, 1), 
        cellSize=1/10, 
        regular=False, 
        qdegree=3 )

x,y = mesh.X


V = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, continuous=True, varsymbol=r"\mathbf{u}")
P = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True, varsymbol=r"P")
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, continuous=True, varsymbol=r"T")

with mesh.access(V,T):
    V.data[:,0] = uw.function.evaluate(x, V.coords)
    V.data[:,1] = uw.function.evaluate(y, V.coords)
    T.data[:,0] = uw.function.evaluate(sympy.sin(2 * sympy.pi * x), T.coords)

# %%
fn0 = V.sym[0] * 3 + x * T.sym[0]
fn0_s = uw.function.expression.substitute(fn0)

# %%
fn1 = V.sym[0] * three + x * T.sym[0]
fn1_s = uw.function.expression.substitute(fn1)

# %%
fn2 = alpha ** 2 * x + rho * sympy.sin(pi * y)
fn2_s = uw.function.expression.substitute(fn2)

# %%
fn3 = V.sym * three
print(uw.function.evaluate(fn3, mesh.data)[::10])

# %%
eta0 = expression(r"{\eta_0}", 1000, "ref viscosity")
C0 = expression(r"{C_0}", 10, "viscosity C0")
eta = eta0 * sympy.exp(-C0 * T.sym[0])

# %%
## Stokes problem / Navier-Stokes problem


navier_stokes = uw.systems.NavierStokes(mesh, velocityField=V, 
                                        pressureField=P, 
                                        order=1,
                                        solver_name="n-navier_stokes")
navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel


# %%
navier_stokes.uf0

# %%
navier_stokes.uF1

# %%
navier_stokes.constitutive_model.viscosity

# %%
navier_stokes.constitutive_model.Parameters.shear_viscosity_0 = eta
navier_stokes.constitutive_model.Parameters.yield_stress = 100


# expression( r"\eta", 1, "Dynamic Viscosity")
navier_stokes.penalty = 0
navier_stokes.saddle_preconditioner = sympy.simplify(1 / (navier_stokes.constitutive_model.viscosity + navier_stokes.penalty))

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

# Surface normals provided by DMPLEX

navier_stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
navier_stokes.add_dirichlet_bc((sympy.oo,0.0), "Bottom")
navier_stokes.add_dirichlet_bc((0.0,sympy.oo), "Left")
navier_stokes.add_dirichlet_bc((0.0,sympy.oo), "Right")

navier_stokes.bodyforce = sympy.Matrix((0, alpha * T.sym[0]))


# %%
navier_stokes.uf0

# %%
navier_stokes.uF1

# %%
navier_stokes.constitutive_model.viscosity

# %%
navier_stokes.Unknowns.E

# %%
navier_stokes.delta_t

# %%
navier_stokes.uF1.subs(navier_stokes.penalty, navier_stokes.penalty.value)

# %%
navier_stokes.uf0.simplify()

# %%
vfm = uw.constitutive_models.ViscousFlowModel(navier_stokes.Unknowns)
vfm.flux

# %%
vpfm = uw.constitutive_models.ViscoPlasticFlowModel(navier_stokes.Unknowns)
vpfm.flux

# %%
vepfm = uw.constitutive_models.ViscoElasticPlasticFlowModel(navier_stokes.Unknowns)
vepfm.flux

# %%
