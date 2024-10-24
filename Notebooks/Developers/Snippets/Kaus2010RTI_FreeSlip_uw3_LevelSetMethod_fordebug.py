#!/usr/bin/env python
# coding: utf-8
# %%

# ### Rayleigh-Taylor instability
# 
# This notebook models the Rayleigh-Taylor instability outlined in Kaus et al. (2010).
# 
# ### References
# Kaus, B. J., Mühlhaus, H., & May, D. A. (2010). A stabilization algorithm for geodynamic numerical simulations with a free surface. Physics of the Earth and Planetary Interiors, 181(1-2), 12-20.

# ### Particle based level set Ex in uw3
# 
# Ex_Stokes_Swarm_RT_Spherical.py

# 
# # Level Set Method
# 
# 
# ### Level set function
# 
# If $\Gamma$ denotes the interface that is to be associated and tracked with a level set function $\phi$, and $\Omega$ is a bounded region.
# 
# $$\left\{\begin{matrix}\begin{align}
# \phi(r,t) = d && (r \in \Omega) \\ 
# \phi(r,t) = -d && (r \notin \Omega) \\ 
# \phi(r,t) = 0 && (r \in \partial \Omega = \Gamma (t))\\  
# \end{align}\end{matrix}\right.$$
# 
# where d is the Euclidian distance to $\Gamma$.
# 
# ### Diffusion Zone
# 
# $$C = \left\{\begin{matrix}\begin{align}
# &C_1     && (\phi \leq 0) \\ 
# &C_2  && (\phi > 0)\\ 
# \end{align}\end{matrix}\right.$$
# 
# 
# $$C = \left\{\begin{matrix}\begin{align}
# &C_1     && (\phi \leq -\alpha h) \\ 
# &C_2  && (\phi \geq \alpha h)\\ 
# &\frac{(C_2-C_1)\phi}{2\alpha h} +\frac{C_1+C_2}{2} && (\left | \phi \right | < \alpha h)\\  
# \end{align}\end{matrix}\right.$$
# 
# 
# ### Advection function $\phi$
# $$\frac{\partial \phi}{\partial t}+\mathbf{v}\cdot \nabla \phi = 0$$
# 
# $$\frac{\partial \phi}{\partial t}+ v_n \left|\nabla \phi \right| = 0 $$
# 
# $$\phi(t+dt) = \phi(t)+(v_x*\frac{d \phi}{dx}+v_y*\frac{d \phi}{dy})*dt$$
# 
# ### Reference
# - Hillebrand, B., Thieulot, C., Geenen, T., Van Den Berg, A. P., & Spakman, W. (2014). Using the level set method in geodynamical modeling of multi-material flows and Earth's free surface. Solid Earth, 5(2), 1087-1098.

# %%


import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from underworld3.cython.petsc_discretisation import petsc_dm_find_labeled_points_local


# %%


#use_diff = True
use_diff = False

if use_diff:
    outputPath = 'op_Kaus2010RTI_FreeSlip_uw3_levelset_diff_test/'
else:
    outputPath = 'op_Kaus2010RTI_FreeSlip_uw3_levelset_test/'
    
if uw.mpi.size == 1:
    # delete previous model run
    if os.path.exists(outputPath):
        for i in os.listdir(outputPath):
            os.remove(outputPath+ i)
            
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)


# %%


u = uw.scaling.units
ndim = uw.scaling.non_dimensionalise
dim = uw.scaling.dimensionalise

# scaling 3: vel
H = 100.  * u.kilometer
velocity     = 1e-9 * u.meter / u.second
g    =   9.81 * u.meter / u.second**2 
bodyforce    = 3300  * u.kilogram / u.metre**3 * g 
mu           = 1e21  * u.pascal * u.second

KL = H
Kt = KL / velocity
KM = mu * KL * Kt

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM

ND_gravity = ndim(9.81 * u.meter / u.second**2)


# %%


render = True   

xres,yres = 50,50 

vdegree,pdegree = 1,0
pcont = False

max_time  = ndim(6.0*u.megayear)
dt_set    = ndim(5e4*u.year)
save_every = 1

xmin, xmax = ndim(-250 * u.kilometer), ndim(250 * u.kilometer)
ymin, ymax = ndim(-500 * u.kilometer), ndim(0 * u.kilometer)
mesh = uw.meshing.StructuredQuadBox(elementRes=(int(xres), int(yres)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))

dx,dy = (xmax-xmin)/xres,(ymax-ymin)/yres

botwall = petsc_dm_find_labeled_points_local(mesh.dm,"Bottom")
topwall = petsc_dm_find_labeled_points_local(mesh.dm,"Top")


# %%


def plot_mesh(title,mesh,showFig=True):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data() 
    pvmesh.clear_point_data()
    
def plot_mesh_var(title,mesh,var,showVar=False,showFig=True):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    if showVar:
        with mesh.access():
            points = np.zeros((mesh.data.shape[0],3))
            points[:,0] = mesh.data[:,0]
            points[:,1] = mesh.data[:,1]
            point_cloud = pv.PolyData(points)
            point_cloud.point_data["M"] = uw.function.evalf(var, mesh.data, mesh.N)
        #pl.add_points(point_cloud, color="red",render_points_as_spheres=False, point_size=3, opacity=0.5)
        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars='M',
                        use_transparency=False, opacity=0.95, point_size= 3)
    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data()
    pvmesh.clear_point_data()
    
def plot_mesh_var2(title,mesh,var,showVar=False,showFig=True):
    import numpy as np
    import pyvista as pv
    import vtk
    
    pv.global_theme.background = "white"
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.jupyter_backend = "static"
    pv.global_theme.smooth_shading = True
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    mesh.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()
    pl.add_mesh(pvmesh,'Black', 'wireframe')

    if showVar:
        with mesh.access():
            points = np.zeros((mesh.data.shape[0],3))
            points[:,0] = mesh.data[:,0]
            points[:,1] = mesh.data[:,1]
            point_cloud = pv.PolyData(points)
            point_cloud.point_data["M"] = var
        #pl.add_points(point_cloud, color="red",render_points_as_spheres=False, point_size=3, opacity=0.5)
        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars='M',
                        use_transparency=False, opacity=0.95, point_size= 3)
    pl.add_title(title,font_size=11)
    if showFig:
        pl.show(cpos="xy")
    
    pl.screenshot(outputPath+title+".png",window_size=pv.global_theme.window_size,return_img=False) 
    pvmesh.clear_data()
    pvmesh.clear_point_data()


# %%


lightIndex = 0
denseIndex = 1
amplitude = ndim(5*u.kilometer)
offset = ndim(-100.*u.kilometer)
wavelength = ndim(500.*u.kilometer)
k = 2.0 * np.pi / wavelength

interfaceSwarm = uw.swarm.Swarm(mesh)
npoints = 101
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), npoints)
y = offset + amplitude * np.cos(k * x)
interface_coords = np.ascontiguousarray(np.array([x,y]).T)
interfaceSwarm.add_particles_with_coordinates(interface_coords)

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=vdegree,continuous=True)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=pdegree,continuous=pcont)
material_mesh = uw.discretisation.MeshVariable("M", mesh, 1, degree=1,continuous=True)
phi = uw.discretisation.MeshVariable("phi", mesh, 1, degree=1,continuous=True, varsymbol=r"\phi")
timeField = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)


# %%


from scipy.spatial import distance
# https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
def points_in_polygon(pts,polygon):
    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask

def init_phi():
    with interfaceSwarm.access():
        index = uw.kdtree.KDTree(interfaceSwarm.data)
        index.build_index()
    indices, dist_sqr, found = index.find_closest_point(mesh.data)
    phi_values = np.sqrt(dist_sqr)
    
    line = mesh.data[topwall]
    x1,y1 = line[:,0],line[:,1]
    zipxy = zip(x1,y1)
    zipxy = sorted(zipxy,reverse=True)
    x2,y2 = zip(*zipxy) 
    line[:,0] = x2 
    line[:,1] = y2
    point_leftwall = np.array([xmin,interface_coords[0,1]])
    point_rightwall = np.array([xmax,interface_coords[0,1]])
    polygon = np.vstack([point_leftwall,interface_coords,point_rightwall])
    polygon = np.concatenate((polygon,line),axis=0)
    polygon = np.vstack([polygon, polygon[0]])
    points = mesh.data
    mask = points_in_polygon(points, polygon)
    
    phi_values[~mask] = -phi_values[~mask]
    
    with mesh.access(phi):
        phi.data[:,0] = phi_values
    return 


# %%
indices = init_phi()


# %%
densityI = ndim(3200 * u.kilogram / u.metre**3)
densityD = ndim(3300 * u.kilogram / u.metre**3)

viscI = ndim(1e20 * u.pascal * u.second)
viscD = ndim(1e21 * u.pascal * u.second)

alphah = 1*dy
def material_parameter_fn(c1,c2,alphah):
    return sympy.Piecewise((c1,phi.sym[0] <= -alphah),
                           (c2,phi.sym[0] > alphah),
                           ((c2-c1)*phi.sym[0]/alphah/2.+(c1+c2)/2, True))

if use_diff:
    visc_fn = material_parameter_fn(viscI,viscD,alphah)
    density_fn = material_parameter_fn(densityI,densityD,alphah) 
    material_fn = sympy.Piecewise((lightIndex,phi.sym[0] < 0.0,),(denseIndex, True))
    material_mesh.sym[0] = material_fn
    print("use_diff")

else:
    visc_fn = sympy.Piecewise((viscD,phi.sym[0] > 0.0),(viscI, True))
    density_fn = sympy.Piecewise((densityD,phi.sym[0] > 0.0),(densityI, True))  
    material_fn = sympy.Piecewise((lightIndex,phi.sym[0] < 0.0,),(denseIndex, True))
    material_mesh.sym[0] = material_fn


# %%
plot_mesh_var("phi_step_init",mesh,phi.fn,showVar=True,showFig=True)


# %%
plot_mesh_var("material_step_init",mesh,material_mesh.sym[0],showVar=True,showFig=True)


# %%

# %%

# %%
## update phi
adv_phi = uw.systems.AdvDiffusion(mesh, u_Field=phi, V_fn=v.sym, solver_name="adv_phi")
adv_phi.constitutive_model = uw.constitutive_models.DiffusionModel
adv_phi.constitutive_model.Parameters.diffusivity= 0.000001
# adv_phi.solve(timestep=0.1)


# %%
V_fn_matrix = mesh.vector.to_matrix(adv_phi.V_fn)
uw.function.evaluate(V_fn_matrix[0,0], mesh.data, simplify=True, verbose=True)

# %%
0/0

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.shear_viscosity_0
stokes.add_dirichlet_bc((0.0,0.0), "Left", (0,))
stokes.add_dirichlet_bc((0.0,0.0), "Right", (0,))
stokes.add_dirichlet_bc((0.0,0.0), "Bottom", (0,1))
stokes.add_dirichlet_bc((0.0,0.0), "Top", (1,))

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-6
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["ksp_atol"] = 1.0e-6
stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_monitor_short"] = None


# %%
### solver test
stokes.solve(zero_init_guess=False)
dt_solver = stokes.estimate_dt()
dt = min(dt_solver,dt_set)

#interfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt,evalf=True)
adv_phi.solve(timestep=dt,zero_init_guess=True)


# %%
0/0




# %%





# %%


# def _adjust_time_units(val):
#     """ Adjust the units used depending on the value """
#     if isinstance(val, u.Quantity):
#         mag = val.to(u.years).magnitude
#     else:
#         val = dim(val, u.years)
#         mag = val.magnitude
#     exponent = int("{0:.3E}".format(mag).split("E")[-1])

#     if exponent >= 9:
#         units = u.gigayear
#     elif exponent >= 6:
#         units = u.megayear
#     elif exponent >= 3:
#         units = u.kiloyears
#     elif exponent >= 0:
#         units = u.years
#     elif exponent > -3:
#         units = u.days
#     elif exponent > -5:
#         units = u.hours
#     elif exponent > -7:
#         units = u.minutes
#     else:
#         units = u.seconds
#     return val.to(units)


# %%


# step      = 0
# max_steps = 2
# time      = 0
# dt        = 0

# w = []
# dwdt = []
# times = []

# while time < max_time:
# #while step < max_steps: 
#     if uw.mpi.rank == 0:
#         string = """Step: {0:5d} Model Time: {1:6.1f} dt: {2:6.1f} ({3})\n""".format(
#         step, _adjust_time_units(time),
#         _adjust_time_units(dt),
#         datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#         sys.stdout.write(string)
#         sys.stdout.flush()
        
#     stokes.solve(zero_init_guess=False)
    
#     if step%save_every ==0:
#         if uw.mpi.rank == 0:
#             print(f'\nSave data:')
#         with mesh.access(timeField):
#             timeField.data[:,0] = dim(time, u.megayear).m
#         mesh.petsc_save_checkpoint(meshVars=[v, p, timeField,material_mesh,phi], index=step, outputPath=outputPath)
#         interfaceSwarm.petsc_save_checkpoint(swarmName='interfaceSwarm', index=step, outputPath=outputPath) 

#     times.append(time)
#     dt_solver = stokes.estimate_dt()
#     dt = min(dt_solver,dt_set)
    
#     interfaceSwarm.advection(V_fn=stokes.u.sym, delta_t=dt,evalf=True)
    
#     #adv_phi.solve(timestep=dt,zero_init_guess=True)
#     advection_phi(stokes.u.sym,dt)

#     step += 1
#     time += dt


# %%


plot_mesh_var("phi_step_final",mesh,phi.sym,showVar=True,showFig=True)


# %%


plot_mesh_var("material_step_final",mesh,material_mesh.sym[0],showVar=True,showFig=True)


# %%


plot_mesh_var("density_step_final",mesh,density_fn,showVar=True,showFig=True)


# %%


with mesh.access():
    dis = dy*2
    dis_data = uw.function.evalf(phi.fn,mesh.data)
    condition = np.abs(dis_data) <=dis
    dis_plot = dis_data[condition]
    nodes_plot = mesh.data[condition]

with interfaceSwarm.access():
    interface_coords = interfaceSwarm.data
    
    
fname = "Mesh nodes closed to the interface"
fig, ax1 = plt.subplots(nrows=1, figsize=(5,5))
ax1.set(xlabel='x coordinates [km]', ylabel='Interface Depth [km]') 
#ax1.set(xlabel='Time [Myrs]', ylabel='Interface Depth [km]') 
ax1.set_title(fname)
ax1.scatter(nodes_plot[:,0]*KL.m,nodes_plot[:,1]*KL.m,c=dis_plot)
ax1.scatter(interface_coords[:,0]*KL.m,interface_coords[:,1]*KL.m,c = "k")
# ax1.set_ylim([-500,-100])
# ax1.set_xlim([0,6])
# ax1.grid()
#ax1.legend(loc = 'lower left',prop = {'size':8})
plt.savefig(outputPath+fname,dpi=150,bbox_inches='tight')

