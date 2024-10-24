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

# %% [markdown]
# [underworldcode/underworld3] Navier Stokes solver - error when running _setup_pointwise_functions() and smoothing still present in self.Unknowns.DuDt (Issue #256)
# underworldcode/underworld3 <underworld3@noreply.github.com>
# ​
# ​
# Subscribed <subscribed@noreply.github.com>
# ​
# When using a Navier Stokes solver with order set to 1, an error occurs when _setup_pointwise_functions() is called (e.g. during solve). To replicate this issue:

# %%
import petsc4py
import underworld3 as uw
import sympy

# %%
resolution = 8
vel     = 1.
order = 1

qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
Pcont = False

# %%
width   = 1.
height  = 1.

# %%
minX, maxX = 0, width
minY, maxY = 0, height

# %%
meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = 1 / resolution, qdegree = qdeg, regular = False)

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=Pdeg, continuous = Pcont)

# %%
navier_stokes = uw.systems.NavierStokesSLCN(
    meshbox,
    velocityField = v_soln,
    pressureField = p_soln,
    rho = 1,
    verbose=True,
    order=order,
)

# %%
navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
navier_stokes.constitutive_model.Parameters.shear_viscosity_0 = 1

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((vel, 0.0), "Top")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")

# %%
# %%
# shows error
navier_stokes.solve(timestep = 0.01, zero_init_guess=True)

# %%

# %%

# %%

# %%
