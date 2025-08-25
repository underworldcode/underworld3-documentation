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
import numpy as np
import sympy
import underworld3 as uw

# %%
res = 0.05
r_o = 1.0
r_int = 0.825
r_i = 0.55

meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*0.5,
                                              cellSize_Outer=res,
                                              centre=False,)


meshbox = uw.meshing.UnstructuredSimplexBox(
    cellSize=res,
    minCoords=(-1.0, -1.0),
    maxCoords=(+1.0, +1.0),
    qdegree=3,
)

mesh = meshball

x, y = mesh.CoordinateSystem.X
r, th = mesh.CoordinateSystem.xR
unit_rvec = mesh.CoordinateSystem.unit_e_0


# Orientation of surface normals
Gamma_N = mesh.Gamma


# %%
rank_var = uw.discretisation.MeshVariable(
    varname="Rank",
    mesh=mesh, 
    vtype = uw.VarType.SCALAR,
    varsymbol=r"Ra",
    continuous=False,
    degree=0,
)


scalar_var = uw.discretisation.MeshVariable(
    varname="Radius",
    mesh=mesh, 
    vtype = uw.VarType.SCALAR,
    varsymbol=r"r"
)

scalar_var.array[...] = uw.function.evaluate(r, scalar_var.coords)
rank_var.array[...] = uw.mpi.rank

# %%
## Functions that we can evaluate anywhere
scalar_fn = scalar_var.sym[0,0] * sympy.sin(x) * sympy.cos(x)

# %%
# Random points in the bounding box of the domain (on each processor)
coords = 1 - 2 * np.random.random(size=(1000,2)) 

# %%
results, is_extrapolated =  uw.function.global_evaluate(scalar_fn, coords, rbf=False, check_extrapolated=True)

# %%
rank, is_extrapolated =  uw.function.global_evaluate(rank_var.sym[0], coords, rbf=False, check_extrapolated=True)

# %%
true_results = uw.function.global_evaluate(r * sympy.sin(x) * sympy.cos(x)  , coords, rbf=False, check_extrapolated=False)

# %%
inside_domain = mesh.points_in_domain(coords)

print(f"{uw.mpi.rank} Max difference              - {(results - true_results).max()}", flush=True)
print(f"{uw.mpi.rank} Mean difference             - {(results - true_results).mean()}", flush=True)

print(f"{uw.mpi.rank} Max difference  [in domain] - {(results - true_results)[inside_domain].max()}", flush=True)
print(f"{uw.mpi.rank} Mean difference [in domain] - {(results - true_results)[inside_domain].mean()}", flush=True)

# %%
swarm = uw.swarm.Swarm(mesh)
error_var = uw.swarm.SwarmVariable("Error", swarm, vtype=uw.VarType.SCALAR, _proxy=False)
s_rank_var = uw.swarm.SwarmVariable("FRank", swarm, vtype=uw.VarType.SCALAR, _proxy=False)

# %%
swarm.add_particles_with_global_coordinates(coords, migrate=False)

# %%
error_var.array[...] = (results - true_results)
s_rank_var.array[...] = rank[...]
uw.mpi.barrier()

# %%
mesh.write_timestep(
    "Parallel_Evaluate",
    meshUpdates=True,
    meshVars=[scalar_var, rank_var],
    outputPath="output",
    index=uw.mpi.size,

)

# %%
swarm.write_timestep(
    "Parallel_Evaluate",
    "data_swarm",
    swarmVars=[error_var, s_rank_var],
    outputPath="output",
    index=uw.mpi.size,
    force_sequential=True,
)
    

# %%
if uw.mpi.rank == 0:
    print("Complete", flush=True)

# %%
exit(0)

# %%
