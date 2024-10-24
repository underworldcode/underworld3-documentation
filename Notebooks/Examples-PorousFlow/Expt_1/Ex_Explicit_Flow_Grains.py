# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Navier Stokes test: flow around a circular inclusion (2D)
#
# No slip conditions
#
# Note ...
#
#
#

# +
import os

os.environ["UW_TIMING_ENABLE"] = "1"

import petsc4py
import underworld3 as uw
from underworld3 import timing

import nest_asyncio
nest_asyncio.apply()

import numpy as np
import sympy

# import psutil
# pid = os.getpid()
# python_process = psutil.Process(pid)
# print(f"Memory usage = {python_process.memory_info().rss//1000000} Mb", flush=True)

# +
# Parameters that define the notebook
# These can be set when launching the script as
# mpirun python3 scriptname -uw_resolution=0.1 etc

resolution = uw.options.getInt("model_resolution", default=20)
refinement = uw.options.getInt("model_refinement", default=0)
model = uw.options.getInt("model_number", default=1)
maxsteps = uw.options.getInt("max_steps", default=201)
restart_step = uw.options.getInt("restart_step", default=-1)

# +
if model == 1:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_{resolution}"

elif model == 2:
    U0 = 0.3
    expt_name = f"NS_benchmark_DFG2d_SLCN_1_ss_{resolution}"

elif model == 3:
    U0 = 1.5
    expt_name = f"NS_benchmark_DFG2d_SLCN_2_{resolution}"

elif model == 4:
    U0 = 3.75
    expt_name = f"NS_test_Re_250_SLCN_{resolution}"

elif model == 5:
    U0 = 15
    expt_name = f"NS_test_Re_1000i_SLCN_{resolution}"
# -

outdir = f"output/output_res_{resolution}"
os.makedirs(".meshes", exist_ok=True)
os.makedirs(f"{outdir}", exist_ok=True)

# +
import pygmsh
from enum import Enum

## NOTE: stop using pygmsh, then we can just define boundary labels ourselves and not second guess pygmsh

class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left  = 4
    inclusion = 5
    All_Boundaries = 1001 

# Mesh a 2D pipe with a circular hole

width = 4.0
height = 1.0
resolution = 10
expt_name = "Expt_2"


csize = 1.0 / resolution
csize_circle = 0.75 * csize
res = csize_circle

width = 4.0
height = 1.0


## Restore inflow samples to inflow points
def pipemesh_return_coords_to_bounds(coords):
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001

    return coords

  
if uw.mpi.rank == 0:
    # Generate local mesh on boss process

    with pygmsh.occ.Geometry() as geom:
        geom.characteristic_length_max = csize

        inclusions = []
        inclusion_curves = []

        rows = 4
        columns = 12
        radius_0 = 0.075
        variation = 0.05


        dy = 1.0/(rows)
        dx = 3.0/(columns)
        
        for row in range(0,rows):
            for col in range(0,columns):

                y = dy*(row+0.5) 
                x = 0.25 + dx * col + ( row%2 ) * 0.5 * dx
                r = radius_0 + variation * np.random.random()

                # inclusion = geom.add_circle(
                #     (x,y,0.0),
                #     radius,
                #     make_surface=False,
                #     mesh_size=csize_circle,
                #     )

                i_points = [
                    geom.add_point([x, y],    csize_circle),
                    geom.add_point([x, y+r], csize_circle),
                    geom.add_point([x-r, y], csize_circle),
                    geom.add_point([x, y-r], csize_circle),
                    geom.add_point([x+r, y], csize_circle),
                ]
                i_quarter_circles = [
                    geom.add_circle_arc(i_points[1], i_points[0], i_points[2]),
                    geom.add_circle_arc(i_points[2], i_points[0], i_points[3]),
                    geom.add_circle_arc(i_points[3], i_points[0], i_points[4]),
                    geom.add_circle_arc(i_points[4], i_points[0], i_points[1]),
                ]

                inclusion_loop = geom.add_curve_loop(i_quarter_circles)
                inclusion = geom.add_plane_surface(inclusion_loop)            

                inclusions.append(inclusion)
                inclusion_curves.append(inclusion_loop)
        
    
        # domain = geom.add_rectangle(
        #     xmin=0.0,
        #     ymin=0.0,
        #     xmax=width,
        #     ymax=height,
        #     z=0,
        #     holes=inclusions,
        #     mesh_size=csize,
        # )



        

        # domain = geom.add_rectangle(
        #          [0,0,0], 
        #          width, 
        #          height,
        #          0)

        corner_points = [
            geom.add_point([0.0, 0.0],  csize),
            geom.add_point([width, 0.0], csize),
            geom.add_point([width, 1.0],  csize),
            geom.add_point([0.0, 1.0], csize),
            ]
        
        bottom, right, top, left = [
            geom.add_line(corner_points[0], corner_points[1]),
            geom.add_line(corner_points[1], corner_points[2]),
            geom.add_line(corner_points[2], corner_points[3]),
            geom.add_line(corner_points[3], corner_points[0]),
        ]

        domain_loop = geom.add_curve_loop((bottom, right, top, left))
        domain = geom.add_plane_surface(domain_loop)
            
        domain_sub = geom.boolean_difference(domain, inclusions)


        geom.synchronize()

        # order determines numbering, b,r,t,l,i (as above)
        geom.add_physical(bottom, label=boundaries.bottom.name)
        geom.add_physical(right, label=boundaries.right.name)
        geom.add_physical(top, label=boundaries.top.name)
        geom.add_physical(left, label=boundaries.left.name)
        
        geom.add_physical(inclusion_curves, label=boundaries.inclusion.name)
        geom.add_physical(domain, label="Elements")
 
        # geom.add_physical(domain.surface.curve_loop.curves[0], label=boundaries.bottom.name)
        # geom.add_physical(domain.surface.curve_loop.curves[1], label=boundaries.right.name)
        # geom.add_physical(domain.surface.curve_loop.curves[2], label=boundaries.top.name)
        # geom.add_physical(domain.surface.curve_loop.curves[3], label=boundaries.left.name)
        # geom.add_physical(inclusion_curves, label=boundaries.inclusion.name)
        # geom.add_physical(domain.surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=0, algorithm=2)
        geom.save_geometry(f".meshes/ns_pipe_flow_{resolution}.msh")

pipemesh = uw.discretisation.Mesh(
    f".meshes/ns_pipe_flow_{resolution}.msh",
    markVertices=True,
    useMultipleTags=True,
    useRegions=True,
    refinement=refinement,
    refinement_callback=None,
    return_coords_to_bounds= pipemesh_return_coords_to_bounds,
    boundaries=boundaries,
    qdegree=3,
)

pipemesh.dm.view()


# Some useful coordinate stuff

x = pipemesh.N.x
y = pipemesh.N.y

Vb = U0


# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
 
    pl = pv.Plotter(window_size=(1000, 250))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
    )

    pl.show(jupyter_backend='html')
# -



v_soln = uw.discretisation.MeshVariable("U", pipemesh, pipemesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", pipemesh, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("Pc", pipemesh, 1, degree=2, continuous=True)
vorticity = uw.discretisation.MeshVariable("omega", pipemesh, 1, degree=1)



# +
passive_swarm = uw.swarm.Swarm(mesh=pipemesh)
passive_swarm.populate(
    fill_param=1,
)

# add new points at the inflow
new_points = 1000
new_coords = np.zeros((new_points,2))
new_coords[:,0] = 0.01
new_coords[:,1] = np.linspace(0, 1.0, new_points)
passive_swarm.add_particles_with_coordinates(new_coords)    

# -
nodal_vorticity_from_v = uw.systems.Projection(pipemesh, vorticity)
nodal_vorticity_from_v.uw_function = sympy.vector.curl(v_soln.fn).dot(pipemesh.N.k)
nodal_vorticity_from_v.smoothing = 1.0e-3
nodal_vorticity_from_v.petsc_options.delValue("ksp_monitor")

# +
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()

stokes = uw.systems.Stokes(
    pipemesh,
    velocityField=v_soln,
    pressureField=p_soln,
    verbose=False,
    solver_name="stokes",
)

stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options.setValue("fieldsplit_velocity_pc_type", "mg")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# # gasm is super-fast ... but mg seems to be bulletproof
# # gamg is toughest wrt viscosity

stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "gamg")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "additive")
stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

# # # mg, multiplicative - very robust ... similar to gamg, additive

# stokes.petsc_options.setValue("fieldsplit_pressure_pc_type", "mg")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_type", "multiplicative")
# stokes.petsc_options.setValue("fieldsplit_pressure_pc_mg_cycle_type", "v")

stokes.tolerance = 0.00001


stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# Constant visc

stokes.penalty = 10
stokes.bodyforce = sympy.Matrix([0, 0])


# Velocity boundary conditions

stokes.add_dirichlet_bc(
    (0.0, 0.0),
    "inclusion",
)

stokes.add_dirichlet_bc((None, 0.0), "top")
stokes.add_dirichlet_bc((None, 0.0), "bottom")
stokes.add_dirichlet_bc((Vb, 0.0), "left")
# -


stokes.solve(zero_init_guess=True)

continuous_pressure_projection = uw.systems.Projection(pipemesh, p_cont)
continuous_pressure_projection.uw_function = p_soln.sym[0]
continuous_pressure_projection.solve()

# +
## Write out data 

# ! mkdir -p Expt_1
# ! cp Ex_Explicit_Flow_Grains.py Expt_1

pipemesh.write_timestep(
    "ExplicitGrains",
    meshUpdates=True,
    meshVars=[p_soln, v_soln],
    outputPath="Expt_2",
    index=0,
)
# -

# ! ls Expt_1/

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Pc"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)

    # pl.add_points(
    #     point_cloud,
    #     color="Black",
    #     render_points_as_spheres=False,
    #     point_size=5,
    #     opacity=0.66,
    # )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="Pc",
        use_transparency=False,
        opacity=1.0,
    )
    
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)
    pl.add_mesh(pvstream)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=5,
        opacity=0.25,
    )

    pl.show(jupyter_backend="html")
# -

0/0

# +
navier_stokes.tolerance = 1.0e-4
navier_stokes.delta_t = 10 # stokes-like at the beginning

if model == 2:  # Steady state !
    # remove the d/dt term ... replace the time dependence with the
    # steady state advective transport term
    # to lean towards steady state solutions

    navier_stokes.UF0 = -(
        navier_stokes.rho * (v_soln.sym - v_soln_1.sym) / navier_stokes.delta_t
    )
# -


navier_stokes.view()

# +
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
navier_stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 2
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

# +
timing.reset()
timing.start()

navier_stokes.solve(
    timestep=10, verbose=False, 
)  # Stokes-like initial flow

nodal_vorticity_from_v.solve()

timing.print_table(display_fraction=0.999)
# -

continuous_pressure_projection.solve()


def plot_V_mesh(filename):

    if uw.mpi.size == 1:
        
        import pyvista as pv
        import underworld3.visualisation as vis
    
        pvmesh = vis.mesh_to_pv_mesh(pipemesh)
        pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
        pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
        pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    
        velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
        velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    
        pl = pv.Plotter(window_size=(1000, 750))
    
        # point sources at cell centres for streamlines
    
        points = np.zeros((pipemesh._centroids.shape[0], 3))
        points[:, 0] = pipemesh._centroids[:, 0]
        points[:, 1] = pipemesh._centroids[:, 1]
        point_cloud = pv.PolyData(points)

        passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)
    
        pvstream = pvmesh.streamlines_from_source(
            point_cloud, vectors="V", integration_direction="forward", max_steps=10, 
        )

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="Omega",
            use_transparency=False,
            opacity=1.0,
            show_scalar_bar=False,
        )

        
        pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], 
                      mag=0.025 / U0, opacity=0.75, 
                      show_scalar_bar=False)
        
        pl.add_mesh(pvstream, show_scalar_bar=False)

        pl.add_points(
            passive_swarm_points,
            color="Black",
            render_points_as_spheres=True,
            point_size=5,
            opacity=0.5,
        )
    
    
        pl.camera.SetPosition(0.75, 0.2, 1.5)
        pl.camera.SetFocalPoint(0.75, 0.2, 0.0)
        pl.camera.SetClippingRange(1.0, 8.0)
    
    
        # pl.camera_position = "xz"
        pl.screenshot(
            filename="{}.png".format(filename),
            window_size=(2560, 1280),
            return_img=False,
        )
    
        pl.clear()

ts = 0
elapsed_time = 0.0
dt_ns = 0.005
delta_t_diff, delta_t_adv  = navier_stokes.estimate_dt()
delta_t = dt_ns


for step in range(0, maxsteps): #1500

    navier_stokes.solve(timestep=delta_t, zero_init_guess=False, verbose=False)

    # update passive swarm
    passive_swarm.advection(v_soln.sym, delta_t, order=2, corrector=False, evalf=False)

    # add new points at the inflow
    npoints = 200
    passive_swarm.dm.addNPoints(npoints)
    with passive_swarm.access(passive_swarm.particle_coordinates):
        for i in range(npoints):
            passive_swarm.particle_coordinates.data[
                -1 : -(npoints + 1) : -1, :
            ] = np.array([0.0, 0.195] + 0.01 * np.random.random((npoints, 2)))

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))

    if ts % 10 == 0:
        nodal_vorticity_from_v.solve()
        plot_V_mesh(filename=f"{outdir}/{expt_name}.{ts:05d}")

        pipemesh.write_timestep(
            expt_name,
            meshUpdates=True,
            meshVars=[p_soln, v_soln, vorticity, St],
            outputPath=outdir,
            index=ts,
        )

        passive_swarm.write_timestep(
            expt_name,
            "passive_swarm",
            swarmVars=None,
            outputPath=outdir,
            index=ts,
            force_sequential=True,
        )

    elapsed_time += delta_t
    ts += 1
# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(pipemesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, v_soln.sym.dot(v_soln.sym))
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, vorticity.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.point_data["Pc"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    ustar = navier_stokes.Unknowns.DuDt.psi_star[0].sym
    pvmesh.point_data["Vs"] = vis.scalar_fn_to_pv_points(pvmesh, ustar.dot(ustar))

    # point sources at cell centres
    points = np.zeros((pipemesh._centroids.shape[0], 3))
    points[:, 0] = pipemesh._centroids[:, 0]
    points[:, 1] = pipemesh._centroids[:, 1]
    point_cloud = pv.PolyData(points)

    passive_swarm_points = uw.visualisation.swarm_to_pv_cloud(passive_swarm)


    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", integration_direction="forward", max_steps=10
    )

    points = vis.swarm_to_pv_cloud(passive_swarm)
    point_cloud = pv.PolyData(points)

    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)

    # pl.add_points(
    #     point_cloud,
    #     color="Black",
    #     render_points_as_spheres=False,
    #     point_size=5,
    #     opacity=0.66,
    # )

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        show_edges=True,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )
    
    pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.025 / U0, opacity=0.75)
    pl.add_mesh(pvstream)

    pl.add_points(
        passive_swarm_points,
        color="Black",
        render_points_as_spheres=True,
        point_size=2,
        opacity=0.25,
    )

    pl.show()
# -



