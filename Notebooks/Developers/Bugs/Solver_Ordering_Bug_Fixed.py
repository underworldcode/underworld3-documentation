# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import underworld3 as uw
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local
import sympy
import numpy as np

mesh1 = uw.meshing.StructuredQuadBox(elementRes=(8, 8), minCoords=(0, -1), maxCoords=(1, 0))
topwall1 = petsc_dm_find_labeled_points_local(mesh1.dm,"Top")

v1 = uw.discretisation.MeshVariable("V", mesh1, mesh1.dim, degree=1,continuous=True)
p1 = uw.discretisation.MeshVariable("P", mesh1, 1, degree=0,continuous=False)
dpdx1 = uw.discretisation.MeshVariable("dpdx", mesh1, vtype=uw.VarType.SCALAR, degree=1, continuous=True, varsymbol=r"\frac{\delta \bar{p}}{\delta x1}")


# order: stokes projection (wrong solution)
bodyf = 405000.
# stokes and projection
stokes1 = uw.systems.Stokes(mesh1, velocityField=v1, pressureField=p1)
stokes1.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes1.bodyforce = sympy.Matrix([0, -1 * bodyf])
stokes1.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes1.saddle_preconditioner = 1.0 / stokes1.constitutive_model.Parameters.shear_viscosity_0
stokes1.add_essential_bc((0.0,None), "Left")
stokes1.add_essential_bc((0.0,None), "Right")
stokes1.add_essential_bc((0.0,0.0), "Bottom")

dpdx_sym = sympy.diff(p1.sym[0],mesh1.X[0])
dpdx_calc1 = uw.systems.Projection(mesh1,dpdx1,degree=1)
dpdx_calc1.smoothing = 1.0e-6
dpdx_calc1.uw_function = stokes1.constitutive_model.flux[1,1]
dpdx_calc1.petsc_options.delValue("ksp_monitor")


# dpdx_calc1._build(verbose=True)
# stokes1._build(verbose=True)


stokes1.solve(zero_init_guess=False)
dpdx_calc1.solve()

# %%
stokes1.dm.view()

# %%
dpdx_calc1.dm.view()
dpdx_calc1.name

# %%
dpdx_calc1.petsc_options_prefix

# %%
v2 = uw.discretisation.MeshVariable("V2", mesh1, mesh1.dim, degree=1,continuous=True)
p2 = uw.discretisation.MeshVariable("P2", mesh1, 1, degree=0,continuous=False)
dpdx2 = uw.discretisation.MeshVariable("dpdx2", mesh1, vtype=uw.VarType.SCALAR, degree=1, continuous=True, varsymbol=r"\frac{\delta \bar{p}}{\delta x2}")

# order: projection stokes (right solution)
dpdx_sym = sympy.diff(p2.sym[0],mesh1.X[0])
dpdx_calc2 = uw.systems.Projection(mesh1,dpdx2,degree =1)
dpdx_calc2.uw_function = dpdx_sym
dpdx_calc2.smoothing = 1.0e-6
dpdx_calc2.petsc_options.delValue("ksp_monitor")

stokes2 = uw.systems.Stokes(mesh1, velocityField=v2, pressureField=p2)
stokes2.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes2.bodyforce = sympy.Matrix([0, -1 * bodyf])
stokes2.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes2.saddle_preconditioner = 1.0 / stokes2.constitutive_model.Parameters.shear_viscosity_0
stokes2.add_essential_bc((0.0,None), "Left")
stokes2.add_essential_bc((0.0,None), "Right")
stokes2.add_essential_bc((0.0,0.0), "Bottom")

stokes2.solve(zero_init_guess=False)
dpdx_calc2.solve()

fn1 = v1.sym[0] 
fn2 = v2.sym[0]  
data1  = uw.function.evaluate(fn1,mesh1.data[topwall1])
data2  = uw.function.evaluate(fn2,mesh1.data[topwall1])
error = data1-data2
print(error)

# %%
with mesh1.access():
    print(v1.data[0:10])    
    print(p1.data[0:10])

# %%
with mesh1.access():
    print(v2.data[0:10])
    print(p2.data[0:10])
