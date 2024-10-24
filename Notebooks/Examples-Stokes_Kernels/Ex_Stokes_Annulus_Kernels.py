# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Cylindrical Stokes 
#
# Mesh with embedded internal surface. This allows us to introduce an internal force integral
#

from IPython.display import IFrame
IFrame(src="./stokes_annulus_plot.html", width=750, height=750)

# to fix visualisation issue
import nest_asyncio
nest_asyncio.apply()

# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

res = 0.025
r_o = 1.0
r_int = 0.825
r_i = 0.55

free_slip_upper = True
free_slip_lower = True

expt_name = "Annulus_Kernel_np8"
output_path = "output"

uw.require_dirs(["output"])
# -
meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=r_o, 
                                              radiusInternal=r_int, 
                                              radiusInner=r_i, 
                                              cellSize_Inner=res,
                                              cellSize_Internal=res*0.5,
                                              cellSize_Outer=res,
                                              centre=False,
                                              filename="tmp_fixedstarsMesh.msh")


meshball.view()

# +
norm_v = uw.discretisation.MeshVariable("N", meshball, 2, degree=1, varsymbol=r"{\hat{n}}")

projection = uw.systems.Vector_Projection(meshball, norm_v)
projection.uw_function = sympy.Matrix([[0,0]])
projection.smoothing = 1.0e-3

# Point in a consistent direction wrt vertical 
GammaNorm = meshball.Gamma.dot(meshball.CoordinateSystem.unit_e_0) / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))

# projection.add_natural_bc((0,0), "All_Edges")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Internal")
projection.add_natural_bc(meshball.Gamma * GammaNorm, "Upper")
if r_i != 0.0:
    projection.add_natural_bc(meshball.Gamma * GammaNorm, "Lower")

projection.solve(verbose=False, debug=False)

with meshball.access(norm_v):
    norm_v.data[:,:] /= np.sqrt(norm_v.data[:,0]**2 + norm_v.data[:,1]**2).reshape(-1,1)



# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

radius_fn = meshball.CoordinateSystem.xR[0]
unit_rvec = meshball.CoordinateSystem.unit_e_0
gravity_fn = 1  # radius_fn / r_o

# Some useful coordinate stuff

x, y = meshball.CoordinateSystem.X
r, th = meshball.CoordinateSystem.xR

# Null space in velocity (constant v_theta) expressed in x,y coordinates
v_theta_fn_xy = r * meshball.CoordinateSystem.rRotN.T * sympy.Matrix((0,1))

Rayleigh = 1.0e5
# -
v_soln = uw.discretisation.MeshVariable("V0", meshball, 2, degree=2, varsymbol=r"{v_0}")
v_soln1 = uw.discretisation.MeshVariable("V1", meshball, 2, degree=2, varsymbol=r"{v_1}")
p_soln = uw.discretisation.MeshVariable("p", meshball, 1, degree=1, continuous=True)
p_cont = uw.discretisation.MeshVariable("pc", meshball, 1, degree=1, continuous=True)


# +
# Create Stokes object

stokes = Stokes(
    meshball, velocityField=v_soln, pressureField=p_soln
)

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = 1.0
stokes.penalty = 1.0

stokes.tolerance = 1.0e-6

stokes.petsc_options.setValue("ksp_monitor", None)
stokes.petsc_options.setValue("snes_monitor", None)
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

t_init = sympy.sin(5*th) * sympy.exp(-1000.0 * ((r - r_int) ** 2)) 



# +
## First solve with known normals

stokes.bodyforce = sympy.Matrix([0,0])
Gamma = meshball.CoordinateSystem.unit_e_0

stokes.add_natural_bc(-t_init * unit_rvec, "Internal")

if free_slip_upper:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
else:
    stokes.add_essential_bc((0.0,0.0), "Upper")

if r_i != 0.0:
    if free_slip_lower:
        stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
    else:
        stokes.add_essential_bc((0.0,0.0), "Lower")

stokes.solve()



# +
# Null space evaluation

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

# print(norm/vnorm, vnorm)

with meshball.access(v_soln):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_soln.coords) / vnorm
    v_soln.data[...] -= dv 

with meshball.access(v_soln1):
    v_soln1.data[...] = v_soln.data[...]

# -

pressure_solver = uw.systems.Projection(meshball, p_cont)
pressure_solver.uw_function = p_soln.sym[0]
pressure_solver.smoothing = 1.0e-6

# +
## Now solve with normals from nodal projection

stokes._reset()

stokes.bodyforce = sympy.Matrix([0,0])
Gamma = meshball.Gamma / sympy.sqrt(meshball.Gamma.dot(meshball.Gamma))
# Gamma = norm_v.sym

stokes.add_natural_bc(-t_init * unit_rvec, "Internal")

if free_slip_upper:
    stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Upper")
else:
    stokes.add_essential_bc((0.0,0.0), "Upper")

if r_i != 0.0:
    if free_slip_lower:    
        stokes.add_natural_bc(10000 * Gamma.dot(v_soln.sym) *  Gamma, "Lower")
    else:
        stokes.add_essential_bc((0.0,0.0), "Lower")

stokes.solve(zero_init_guess=True)

# +
# Null space evaluation

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()

with meshball.access(v_soln):
    dv = uw.function.evaluate(norm * v_theta_fn_xy, v_soln.coords) / vnorm
    v_soln.data[...] -= dv 

# print(norm/vnorm, vnorm)
# -9.662093930530614e-09 0.024291704747453444

norm = I0.evaluate()
# -

I0 = uw.maths.Integral(meshball, v_theta_fn_xy.dot(v_soln.sym))
norm = I0.evaluate()

# +
# Pressure at mesh nodes
pressure_solver.solve()

pstats1 = p_cont.stats()
pstats0 = p_soln.stats()

if uw.mpi.rank == 0:
    print(f"Pressure (C1): {pstats1}")
    print(f"Pressure (C0): {pstats0}")
    print(f"Velocity: {vnorm}")
# -

meshball.write_timestep(
    expt_name,
    meshUpdates=True,
    meshVars=[p_soln, v_soln, p_cont],
    outputPath=output_path,
    index=0,
)

# ls output/Annulus_Kernel_np8.*

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshball)
    
    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)
    velocity_points.point_data["V0"] = vis.vector_fn_to_pv_points(velocity_points, v_soln1.sym)
    velocity_points.point_data["dV"] = velocity_points.point_data["V"] - velocity_points.point_data["V0"]

    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_cont.sym)
    pvmesh.point_data["T"] = vis.scalar_fn_to_pv_points(pvmesh, t_init)
    pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(pvmesh, v_soln1.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["dV"] = pvmesh.point_data["V"] - pvmesh.point_data["V0"]
    pvmesh.point_data["Vmag"] = np.hypot(pvmesh.point_data["V"][:,0],pvmesh.point_data["V"][:,1])

    skip = 6
    points = np.zeros((meshball._centroids[::skip].shape[0], 3))
    points[:, 0] = meshball._centroids[::skip, 0]
    points[:, 1] = meshball._centroids[::skip, 1]
    point_cloud = pv.PolyData(points)

    pvstream = pvmesh.streamlines_from_source(
        point_cloud, vectors="V", 
        integration_direction="both", 
        integrator_type=45,
        surface_streamlines=True,
        initial_step_length=0.01,
        max_time=0.25,
        max_steps=500, 
    )
   

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Grey",
        edge_opacity=0.33,
        scalars="P",
        show_edges=True,
        use_transparency=False,
        opacity=1.0,
        show_scalar_bar=False
    )

    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=4, opacity=0.25)
    #pl.add_arrows(velocity_points.points, velocity_points.point_data["V0"], mag=4, color="Black")
    # pl.add_arrows(velocity_points.points, velocity_points.point_data["dV"], mag=100, color="Black")
    pl.add_mesh(pvstream, opacity=0.3, show_scalar_bar=False, cmap="Greens", render_lines_as_tubes=False)

    
    vsol_rms = np.sqrt(velocity_points.point_data["V"][:, 0] ** 2 + velocity_points.point_data["V"][:, 1] ** 2).mean()
    # print(vsol_rms)

    pl.export_html("stokes_annulus_plot.html")
    # pl.show(cpos="xy", jupyter_backend="trame")


# -
if uw.mpi.size == 1:
    from IPython.display import IFrame
    IFrame(src="./stokes_annulus_plot.html", width=750, height=750)


stokes.view()


