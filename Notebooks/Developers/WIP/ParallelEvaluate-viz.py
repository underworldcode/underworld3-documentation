# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python (Pixi)
#     language: python
#     name: pixi-kernel-python3
# ---

# %%
# This is required to fix pyvista
# (visualisation) crashes in interactive notebooks (including on binder)

import nest_asyncio
nest_asyncio.apply()

# %%
import numpy as np
import sympy
import underworld3 as uw

# %%
index = 12
basename = "output/Parallel_Evaluate"

# %%
meshball = uw.discretisation.Mesh(f"{basename}.mesh.{index:05d}.h5")

# meshball.view()

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR
unit_rvec = meshball.CoordinateSystem.unit_e_0


# Orientation of surface normals
Gamma_N = meshball.Gamma


# %%
scalar_var = uw.discretisation.MeshVariable(
    varname="Radius", mesh=meshball, vtype=uw.VarType.SCALAR, varsymbol=r"r"
)

rank_var = uw.discretisation.MeshVariable(
    varname="Rank",
    mesh=meshball,
    vtype=uw.VarType.SCALAR,
    varsymbol=r"Ra",
    degree=0,
)

# %%
scalar_var.read_timestep(f"{basename}", "Radius", index, verbose=True)
rank_var.read_timestep(f"{basename}", "Rank", index, verbose=True)

# %%

# %%
swarm = uw.swarm.Swarm(meshball)

error_var = uw.swarm.SwarmVariable(
    "Error",
    swarm,
    vtype=uw.VarType.SCALAR,
    _proxy=False,
)

s_rank_var = uw.swarm.SwarmVariable(
    "FRank",
    swarm,
    vtype=uw.VarType.SCALAR,
    _proxy=False,
)


swarm.read_timestep(
    basename,
    "data_swarm",
    index,
    migrate=False,
)

error_var.read_timestep(
    basename,
    "data_swarm",
    "Error",
    index,
)

s_rank_var.read_timestep(
    basename,
    "data_swarm",
    "FRank",
    index,
)



# %%
in_or_out = meshball.points_in_domain(swarm.particle_coordinates.array[...].reshape(-1,2))

# %%

# %%
import pyvista as pv

pvmesh = uw.visualisation.mesh_to_pv_mesh(meshball)

pvmesh.cell_data["rank"] = uw.visualisation.scalar_fn_to_pv_points(
    pvmesh.cell_centers(), rank_var.sym
)

error_swarm = uw.visualisation.swarm_to_pv_cloud(swarm)
error_swarm.point_data["E"] = np.sqrt(error_var.array[:, 0, 0]**2)
error_swarm.point_data["E2"] = np.sqrt(error_var.array[:, 0, 0]**2) * in_or_out.astype(float)
error_swarm.point_data["R"] = np.sqrt(s_rank_var.array[:, 0, 0]**2) * in_or_out.astype(float)



plotter = pv.Plotter()

plotter.add_mesh(pvmesh, scalars="rank", cmap="rainbow", show_edges=True, opacity=0.2)
plotter.add_points(error_swarm, cmap="rainbow", scalars="R", point_size=4)
# plotter.add_points(error_swarm,  scalars="E", point_size=3)

plotter.show()

# %%
swarm.dm.getLocalSize()

# %%
