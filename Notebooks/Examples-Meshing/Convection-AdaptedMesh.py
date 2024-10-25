# %%
# +
## Mesh refinement ...

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os

os.environ["UW_TIMING_ENABLE"] = "1"
os.environ["SYMPY_USE_CACHE"] = "no"

import petsc4py
from petsc4py import PETSc

from underworld3 import timing
from underworld3 import adaptivity

import underworld3 as uw
from underworld3 import function
from enum import Enum

import numpy as np
import sympy


# %%
class bd(Enum):
    Upper=2
    
batmesh = uw.discretisation.Mesh(
    "Batmesh_number_one.h5",
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CYLINDRICAL2D,
    boundaries=bd)


# %%
batmesh.dm.view()

# %%
Rayleigh_number = uw.function.expression(r"\textrm{Ra}", sympy.sympify(10)**6, "Rayleigh number")

# %%
v_soln = uw.discretisation.MeshVariable("U", batmesh, batmesh.dim, degree=2, continuous=True)
p_soln = uw.discretisation.MeshVariable("P", batmesh, 1, degree=1, continuous=True)
t_soln = uw.discretisation.MeshVariable("T", batmesh, 1, degree=2)
cell_properties = uw.discretisation.MeshVariable("Bat", batmesh, 1, degree=0)


# %%
cell_properties.read_timestep(
        "Batmesh_cell_properties",
         data_name="L",
         index=0,
)

# %%
# Create Stokes object

stokes = uw.systems.Stokes(
    batmesh,
    velocityField=v_soln,
    pressureField=p_soln,
)

# Constant viscosity

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
stokes.tolerance = 1.0e-3

unit_r_vec = batmesh.CoordinateSystem.unit_e_0

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.

stokes.add_natural_bc(10000 * unit_r_vec.dot(v_soln.sym) * unit_r_vec, "Upper")
stokes.bodyforce = Rayleigh_number * unit_r_vec * t_soln.sym[0] 

# Make the bat stagnant
# stokes.bodyforce -= 10000 * (1-cell_properties.sym[0]) * v_soln.sym

stokes.view()

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    batmesh,
    u_Field=t_soln,
    V_fn=v_soln,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = 1
adv_diff.f = 10 * (1-cell_properties.sym[0])

adv_diff.add_dirichlet_bc(0.0, "Upper")

t_init = 1-cell_properties.sym[0]

with batmesh.access(t_soln):
    t_soln.data[:,0] = uw.function.evalf(t_init, t_soln.coords)


# %%
stokes.solve(zero_init_guess=True)

# %%
v_soln.view()

# %%
t_step = 0

# %%

for step in range(0, 50):
    stokes.solve(zero_init_guess=False)
    delta_t = 2.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))

    batmesh.write_timestep(
        "Batmesh",
        meshUpdates=True,
        meshVars=[p_soln, v_soln, t_soln],
        outputPath="output",
        index=t_step,
    )

    with batmesh.access(t_soln):
        t_soln.data[:,0] = np.maximum(t_soln.data[:,0], uw.function.evalf(t_init, t_soln.coords))

    t_step += 1


# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(batmesh)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_soln.sym[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["L"] = vis.scalar_fn_to_pv_points(pvmesh, cell_properties.sym[0])

    pl = pv.Plotter(window_size=(750, 750))

    pvstream = pvmesh.streamlines_from_source(
        pvmesh.cell_centers(), 
        vectors="V", 
        integrator_type=45,
        surface_streamlines=True, 
        max_steps=1000,
        max_time=0.25,
    )

    pl.add_mesh(
                pvmesh,
                cmap="Oranges",
                scalars="T",
                edge_color="Grey",
                show_edges=False,
                use_transparency=False,
                opacity=1.0,
                show_scalar_bar=True,
                clim=[0,1]
               )
    
    pl.add_mesh(
                pvmesh,
                scalars="L",
                cmap="Greys_r",
                edge_color="Black",
                opacity="L",
                show_edges=False,
                use_transparency=True,
                show_scalar_bar=False,
                
               )

    # pl.add_mesh(pvstream, cmap="jet", show_scalar_bar=False)

    

    # pl.add_arrows(pvmesh.points, pvmesh.point_data["V"],
    #               mag=float(20.0/Rayleigh_number.sym),
    #              show_scalar_bar=False)


    pl.show(jupyter_backend='html')

# %%
## Check the results saved to file

expt_name = "Batmesh"
output_path="output"
step = 20

v_soln.read_timestep(expt_name, "U", step, outputPath=output_path)
p_soln.read_timestep(expt_name, "P", step, outputPath=output_path)
t_soln.read_timestep(expt_name, "T", step, outputPath=output_path)



# %%
adv_diff

# %%
