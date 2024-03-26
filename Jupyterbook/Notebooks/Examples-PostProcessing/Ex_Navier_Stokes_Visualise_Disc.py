# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Navier Stokes test: flow in an annulus with a moving boundary (2D)
#

# to fix trame issue
import nest_asyncio
nest_asyncio.apply()

import os
import petsc4py
import underworld3 as uw
import numpy as np
import sympy

# + language="sh"
#
# ls -tr /Users/lmoresi/+Simulations/NS_benchmarks/
# #ls -tr /Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes/output_res_033/*mesh*h5 | tail -10
#
# -


# ls -tr /Users/lmoresi/+Simulations/NS_benchmarks/NS_Annulus_Re1000/Cylinder_NS_rho_1000_25*omega* | tail -5

# +
## Reading the checkpoints back in ... 

step = 320

checkpoint_dir = "/Users/lmoresi/+Simulations/NS_benchmarks/NS_Annulus_Re1000"
# checkpoint_dir = "/Users/lmoresi/+Underworld/underworld3/JupyterBook/Notebooks/Examples-NavierStokes//Users/lmoresi/+Simulations/NS_benchmarks/NS_BMK_DvDt_std"

checkpoint_base = "Cylinder_NS_rho_1000_25"
base_filename = os.path.join(checkpoint_dir, checkpoint_base)

# +
# mesh = uw.discretisation.Mesh(f"{base_filename}.mesh.{step:05d}.h5")
mesh = uw.discretisation.Mesh(f"{base_filename}.mesh.00000.h5")

v_soln_ckpt = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p_soln_ckpt = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
vorticity_ckpt = uw.discretisation.MeshVariable("omega", mesh, 1, degree=1)

passive_swarm_ckpt = uw.swarm.Swarm(mesh)
active_swarm_ckpt = uw.swarm.Swarm(mesh)
# -




# +
v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir)
p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir)
vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir)

# This one is just the individual points
passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym[0])
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity_ckpt.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_soln_ckpt.sym.dot(v_soln_ckpt.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln_ckpt)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)
    
    x,y = mesh.X
    U0 = 1.5
    Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2
        
    # swarm points
    points = vis.swarm_to_pv_cloud(passive_swarm_ckpt)
    swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres
    skip = 5
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5,
    )

    pl = pv.Plotter(window_size=(1000, 750))
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.03, opacity=0.5)

    pl.add_mesh(
        pvmesh,
        cmap="bwr",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        clim=[-1000,1000],
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.25
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.25)
    pl.add_mesh(pvstream, opacity=0.5)

    pl.remove_scalar_bar("Omega")
    pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")
        
    pl.camera.SetPosition(0.0, 0.0, 4.7)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
        
        
    pl.screenshot(
            filename=f"{base_filename}.{step}.png",
            window_size=(1600, 1600),
            return_img=False,
        )
    
    pl.show(jupyter_backend="client")
# -


0/0

# +
import glob
steps = []
U_files = glob.glob(f"{checkpoint_dir}/{checkpoint_base}.mesh.U*h5")
for Uf in U_files:
    steps.append(int(Uf.split('.U.')[1].split('.')[0]))
steps.sort()

print(steps)


# +
# Override output range (but need to run above cell to get the files themselves)

# steps = range(0,100,5)

# +
if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis
    
    pl = pv.Plotter(window_size=(1000, 750))

for step in steps:
    # check the mesh if in a notebook / serial
    
    v_soln_ckpt.read_timestep(checkpoint_base, "U", step, outputPath=checkpoint_dir, verbose=True)
    p_soln_ckpt.read_timestep(checkpoint_base, "P", step, outputPath=checkpoint_dir, verbose=True)
    vorticity_ckpt.read_timestep(checkpoint_base, "omega", step, outputPath=checkpoint_dir, verbose=True)

# This one is just the individual points
    passive_swarm_ckpt = uw.swarm.Swarm(mesh)
    passive_swarm_ckpt.read_timestep(checkpoint_base, "passive_swarm", step, outputPath=checkpoint_dir)

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln_ckpt.sym)
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity_ckpt.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln_ckpt.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, sympy.sqrt(v_soln_ckpt.sym.dot(v_soln_ckpt.sym)))

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln_ckpt)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln_ckpt.sym)
        
    x,y = mesh.X
    U0 = 1.5
    Vb = (4.0 * U0 * y * (0.41 - y)) / 0.41**2

    # swarm points
    points = vis.swarm_to_pv_cloud(passive_swarm_ckpt)
    swarm_point_cloud = pv.PolyData(points)

    # point sources at cell centres
    skip = 15
    points = np.zeros((mesh._centroids[::skip].shape[0], 3))
    points[:, 0] = mesh._centroids[::skip, 0]
    points[:, 1] = mesh._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integration_direction="both",
        max_time=0.5,
    )

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.03, opacity=0.5)


    pl.add_points(swarm_point_cloud, color="Black",
                  render_points_as_spheres=True,
                  point_size=3, opacity=0.5
                )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.05)
    
    pl.add_mesh(
        pvmesh,
        cmap="bwr",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        clim=[-250,250],
        use_transparency=False,
        opacity=0.75,
    )
    
    pl.add_mesh(pvstream, opacity=0.5)


    pl.remove_scalar_bar("Omega")
    # pl.remove_scalar_bar("mag")
    pl.remove_scalar_bar("V")
    
    pl.camera.SetPosition(0.0, 0.0, 4.7)
    pl.camera.SetFocalPoint(0.0, 0.0, 0.0)
    pl.camera.SetClippingRange(1.0, 8.0)
         
    pl.screenshot(
            filename=f"{base_filename}.{step}.png",
            window_size=(1600, 1600),
            return_img=False,
        )
    
    pl.clear()
# -
# ! open .



