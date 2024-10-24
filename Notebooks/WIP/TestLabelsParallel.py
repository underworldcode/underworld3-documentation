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
# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

# %%
import underworld3 as uw
from underworld3.systems import Stokes
import sympy

# %%

mesh1 = uw.meshing.StructuredQuadBox(elementRes=(20, 20), 
                                     minCoords=(0., 0,),
                                     maxCoords=(1., 1.))

x,y = mesh1.X

mesh1.view()

v_soln = uw.discretisation.MeshVariable(r"u", mesh1, 2, degree=2)
p_soln = uw.discretisation.MeshVariable(r"p", mesh1, 1, degree=1, continuous=True)

# Create Stokes object
stokes = Stokes(mesh1, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.saddle_preconditioner = 1.0

# +
# bc's
t_init = sympy.cos(3*x*sympy.pi) * sympy.exp(-1000.0 * ((y - 0.99) ** 2)) 

# stokes.add_essential_bc(sympy.Matrix([1.0, 0.0]), "Top")

stokes.add_natural_bc(sympy.Matrix([0.0, -t_init]), "Top")
stokes.add_essential_bc(sympy.Matrix([sympy.oo, 0.0]), "Bottom")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Left")
stokes.add_essential_bc(sympy.Matrix([0.0,sympy.oo]), "Right")
# -

stokes.bodyforce = sympy.Matrix([0, 0])

print(f'rank: {uw.mpi.rank}, min coord: {mesh1.data[:,0].min(), mesh1.data[:,1].min()}', flush=True)
print(f'rank: {uw.mpi.rank}, max coord: {mesh1.data[:,0].max(), mesh1.data[:,1].max()}', flush=True)

uw.mpi.barrier()

# stokes.petsc_options["pc_type"] = "lu"
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
stokes.tolerance = 1e-6
stokes.solve(verbose=False)


# %%
stats = p_soln.stats()
if uw.mpi.rank == 0:
    print(stats)

# %% [markdown]
# # output
#
#     20x20 quad
#     (441, -4.739496872564774e-10, -0.9070855410329308, 0.907085542103707, -2.0901181208010655e-07, 3.84583413217208, 0.18313495867486096)

# %%
if uw.mpi.size == 1:

    from underworld3 import visualisation as vis
    import numpy as np
    import pyvista as pv

    pvmesh = vis.mesh_to_pv_mesh(mesh1)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym[0])
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)

    pvmesh_v = vis.meshVariable_to_pv_mesh_object(v_soln, alpha=None)
    pvmesh_v.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh_v, v_soln.sym)


    # # point sources at cell centres
    # skip = 19
    # points = np.zeros((mesh1._centroids[::skip].shape[0], 3))
    # points[:, 0] = mesh1._centroids[::skip, 0]
    # points[:, 1] = mesh1._centroids[::skip, 1]
    # point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=[750, 750])

    pl.add_mesh(
        pvmesh,
        cmap="seismic",
        edge_color="Black",
        edge_opacity=0.1,
        show_edges=False,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False
    )

    pl.add_arrows(pvmesh_v.points, pvmesh_v.point_data["V"], mag=2)

    pl.show()


    

# %%
from petsc4py import PETSc


# %%
PETSc.COMM_WORLD

# %%
new_is = PETSc.IS().create()

# %%
new_is.getLocalSize()

# %%
# new_is.setType()

# %%

# %%
lll = mesh1.dm.getLabel("depth")
lll.getNumValues()

# %%
label_is = lll.getStratumIS(2)

# %%
label_is.view()

# %%
mesh1.dm.view()

# %%
