# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw3-venv-run
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Navier Stokes benchmark: 2D pipe flow
# By: Juan Carlos Graciosa 
# &nbsp;
# &nbsp;
#
# References:  
# - https://www.fifty2.eu/innovation/planar-poiseuille-flow-2-d-in-preonlab/ 
# - Cengel, Y. A. (2010). Fluid Mechanics: Fundamentals and Applications (SI Units). Tata McGraw Hill Education Private Limited.
#

# %%
import os

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy
import argparse
import pickle

# parser = argparse.ArgumentParser()
# parser.add_argument('-i', "--idx", type=int, required=True)
# parser.add_argument('-p', "--prev", type=int, required=True) # set to 0 if no prev_res, 1 if there is
# args = parser.parse_args()

# idx = args.idx
# prev = args.prev

idx = 0
prev = 0


# %%
resolution = 16
refinement = 0
save_every = 200
maxsteps = 1
Cmax = 1            # target Courant number

order = 1           # solver order
tol = 1e-12         # solver tolerance (sets atol and rtol)

use_dim = True      # True if using dimensionalised values; False otherwise
case_num = 1

mesh_type = "Pirr" # or Preg, Pirr, Quad
qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
Pcont = False

# %%
expt_name = f"Pois-res{resolution}-order{order}-{mesh_type}-case{case_num}"

outfile = f"{expt_name}_run{idx}"
outdir = f"/scratch/el06/jg0883/Poiseuille/{expt_name}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"{expt_name}_run{prev_idx}"

if uw.mpi.rank == 0 and uw.mpi.size > 1:
    os.makedirs(f"{outdir}", exist_ok=True)

# %%
# dimensionalized values of problem parameters
# from reference

# velocity - m/s
# fluid_rho - kg/m^3
# dynamic viscosity - Pa.s
if case_num == 1:       # Re = 10
    vel_dim         = 0.034
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 2:     # Re = 100
    vel_dim         = 0.34
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 3:     # Re = 1000
    vel_dim         = 3.4
    fluid_rho_dim   = 910
    dyn_visc_dim    = 0.3094
elif case_num == 4:     # Re = 10
    vel_dim         = 1.
    fluid_rho_dim   = 100
    dyn_visc_dim    = 1
elif case_num == 5:     # Re = 100
    vel_dim         = 1
    fluid_rho_dim   = 100
    dyn_visc_dim    = 0.1
elif case_num == 6:     # Re = 1000
    vel_dim         = 1
    fluid_rho_dim   = 100
    dyn_visc_dim    = 0.01

height_dim  = 2 * 0.05          # meters
if case_num in [2,3, 5, 6]:          # Re = 1000
    width_dim   = 10 * height_dim    # meters
else:
    #width_dim   = 6 * height_dim    # meters
    width_dim   = 8 * height_dim    # meters

kin_visc_dim  = dyn_visc_dim / fluid_rho_dim
Re_num        = fluid_rho_dim * vel_dim * height_dim / dyn_visc_dim
if uw.mpi.rank == 0:
    print(f"Reynold's number: {Re_num}")
    print(f"Dimensionalized velocity: {vel_dim}")
    print(f"Dimensionalized height: {height_dim}")
    print(f"Dimensionalized width: {width_dim}")

# %%
if use_dim:
    height  = height_dim
    width   = width_dim

    vel     = vel_dim

    fluid_rho   = fluid_rho_dim
    kin_visc    = kin_visc_dim
    dyn_visc    = dyn_visc_dim
else:
    pass # perform non-dimensionalization here

# %%
minX, maxX = -0.5 * width, 0.5 * width
minY, maxY = -0.5 * height, 0.5 * height

if uw.mpi.rank == 0:
    print("min X, max X:", minX, maxX)
    print("min Y, max Y:", minY, maxY)
    print("kinematic viscosity: ", kin_visc)
    print("fluid density: ", fluid_rho)
    print("dynamic viscosity: ", kin_visc * fluid_rho)

# %%
# cell size calculation
if mesh_type == "Preg":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = height / resolution, qdegree = qdeg, regular = True)
elif mesh_type == "Pirr":
    meshbox = uw.meshing.UnstructuredSimplexBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize = height / resolution, qdegree = qdeg, regular = False)
elif mesh_type == "Quad":
    meshbox = uw.meshing.StructuredQuadBox( minCoords=(minX, minY), maxCoords=(maxX, maxY), elementRes = ((width/height) * resolution, resolution), qdegree = qdeg, regular = False)

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshbox)

    pl = pv.Plotter(window_size=(1000, 750))

    # point sources at cell centres for streamlines

    points = np.zeros((meshbox._centroids.shape[0], 3))
    points[:, 0] = meshbox._centroids[:, 0]
    points[:, 1] = meshbox._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    pl.add_mesh(pvmesh,
                edge_color="Black",
                show_edges=True,
                show_scalar_bar=False)

    pl.show()

# %%
meshbox.dm.view()

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Pdeg, continuous = Pcont)

# %%
if infile is None:
    pass
else:
    if uw.mpi.rank == 0:
        print(f"Reading: {infile}")

    v_soln.read_timestep(data_filename = infile, data_name = "U", index = maxsteps, outputPath = outdir)
    p_soln.read_timestep(data_filename = infile, data_name = "P", index = maxsteps, outputPath = outdir)

# %%
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

navier_stokes = uw.systems.NavierStokesSLCN(
    meshbox,
    velocityField = v_soln,
    pressureField = p_soln,
    rho = fluid_rho,
    verbose=False,
    order=order,
)

navier_stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
# Constant visc
navier_stokes.constitutive_model.Parameters.viscosity = dyn_visc

navier_stokes.penalty = 0
navier_stokes.bodyforce = sympy.Matrix([0, 0])

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc((vel, 0.0), "Left")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Bottom")
navier_stokes.add_dirichlet_bc((0.0, 0.0), "Top")
# right is open

navier_stokes.tolerance = tol

# %%
# navier_stokes.petsc_options["snes_monitor"] = None
# navier_stokes.petsc_options["snes_converged_reason"] = None
# navier_stokes.petsc_options["snes_monitor_short"] = None
# navier_stokes.petsc_options["ksp_monitor"] = None

# navier_stokes.petsc_options["snes_type"] = "newtonls"
# navier_stokes.petsc_options["ksp_type"] = "fgmres"

# navier_stokes.petsc_options["snes_max_it"] = 50
# navier_stokes.petsc_options["ksp_max_it"] = 50

navier_stokes.petsc_options["snes_monitor"] = None
navier_stokes.petsc_options["ksp_monitor"] = None

navier_stokes.petsc_options["snes_type"] = "newtonls"
navier_stokes.petsc_options["ksp_type"] = "fgmres"

navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
navier_stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")

navier_stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
navier_stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
# navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
navier_stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# %%
# set the timestep
# for now, set it to be constant
delta_x = meshbox.get_min_radius()
max_vel = vel

delta_t = Cmax*delta_x/max_vel

if uw.mpi.rank == 0:
    print(f"Min radius: {delta_x}")
    print("Timestep used:", delta_t)


# %%
ts = 0
timeVal =  np.zeros(maxsteps + 1)*np.nan      # time values
elapsed_time = 0.0

# %%
for step in range(0, maxsteps):

    if uw.mpi.rank == 0:
        print(f"Timestep: {step}")

    navier_stokes.solve(timestep = delta_t, zero_init_guess=True)

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))

    if ts % save_every == 0 and ts > 0:
        meshbox.write_timestep(
            outfile,
            meshUpdates=True,
            meshVars=[p_soln, v_soln],
            outputPath=outdir,
            index =ts,
        )

        with open(outdir + f"/{outfile}.pkl", "wb") as f:
            pickle.dump([timeVal], f)

    # update timestep
    ts += 1

# save after all iterations
meshbox.write_timestep(
    outfile,
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath=outdir,
    index =maxsteps,
)

with open(outdir + f"/{outfile}.pkl", "wb") as f:
    pickle.dump([timeVal], f)
