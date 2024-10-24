import underworld3 as uw
import numpy as np
import sympy
import math
import pytest

# ### Test Semi-Lagrangian method in advecting vector fields
# ### Scalar field advection together diffusion tested in a different pytest

# +
# Set up variables of the model
# +
res = 8
velocity = 1 

# ### mesh coordinates
xmin, xmax = 0., 2.
ymin, ymax = 0., 1.

sdev = 0.1
x0 = 0.5
y0 = 0.5

# ### Set up the mesh
### Quads
meshStructuredQuadBox = uw.meshing.StructuredQuadBox(
    elementRes=(int(res*xmax), int(res)),
    minCoords=(xmin, ymin),
    maxCoords=(xmax, ymax),
    qdegree=3,
)

unstructured_simplex_box_irregular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(xmin,ymin), 
    maxCoords=(xmax,ymax),
    cellSize=1 / res, regular=False, qdegree=3, refinement=0
)

unstructured_simplex_box_regular = uw.meshing.UnstructuredSimplexBox(
    minCoords=(xmin,ymin), 
    maxCoords=(xmax,ymax),
    cellSize=1 / res, regular=True, qdegree=3, refinement=0
)

mesh = unstructured_simplex_box_regular


# +
# Create mesh vars
Vdeg        = 2
sl_order    = 1

v           = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=Vdeg)
vect_test   = uw.discretisation.MeshVariable("Vn", mesh, mesh.dim, degree=Vdeg)
vect_anal   = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=Vdeg)
omega       = uw.discretisation.MeshVariable("omega", mesh, 1, degree=2, )

# #### Create the SL object
DuDt = uw.systems.ddt.SemiLagrangian(
                                        mesh,
                                        vect_test.sym,
                                        v.sym,
                                        vtype = uw.VarType.VECTOR,
                                        degree = Vdeg,
                                        continuous = vect_test.continuous,
                                        varsymbol = vect_test.symbol,
                                        verbose = False,
                                        bcs = None,
                                        order = sl_order,
                                        smoothing = 0.0,
                                    )



# +
# ### Set up:
# - Velocity field
# - Initial vector distribution

with mesh.access(v):
    v.data[:, 0] = velocity

x,y = mesh.X

## Irrotational vortex

def v_irrotational(alpha,x0,y0, coords):
    '''
    Irrotational vortex 
    
    $$ (vx, vy) = (-\alpha y r^{-2}, \alpha x r^{-2} $$
    '''

    ar2 = alpha / ((x - x0)**2 + (y - y0)**2 + 0.001) 
    return uw.function.evalf(sympy.Matrix([-ar2 * (y-y0), ar2 * (x-x0)]) ,coords)

    
def v_rigid_body(alpha, x0, y0, coords):
    '''
    Rigid body vortex (with Gaussian envelope)
    
    $$ (vx, vy) = (-\Omega y, \Omega y) $$
    '''
    ar2 = sympy.exp(-alpha*((x - x0)**2 + (y - y0)**2 + 0.000001)) 
    return uw.function.evalf(sympy.Matrix([-ar2 * (y-y0), ar2 * (x-x0)]) ,coords)


with mesh.access(vect_test):
    vect_test.data[:, :] = v_rigid_body(33, x0, y0, vect_test.coords)
    # vect_test.data[:, :] =  v_irrotational(0.01, x0, y0, vect_test.coords)

with mesh.access(vect_anal):
    vect_anal.data[:, :] = v_rigid_body(33, x0+1, y0, vect_anal.coords)
    # vect_anal.data[:, :] = v_irrotational(0.01, x0+1, y0, vect_test.coords)


vorticity_calculator = uw.systems.Projection(mesh, omega)
vorticity_calculator.uw_function = mesh.vector.curl(vect_test.sym)
vorticity_calculator.petsc_options["snes_monitor"]= None
vorticity_calculator.petsc_options["ksp_monitor"] = None



# +
model_time = 0.0
dt = 0.2
max_time = 1.0
step = 0

while step < 5:
    DuDt.update_pre_solve(dt, verbose = False, evalf = True)

    with mesh.access(vect_test): # update vector field
        vect_test.data[...] = DuDt.psi_star[0].data[...]

    model_time += dt
    print(f"{step}: Time - {model_time}")
    step += 1

# -

0/0

# +
vorticity_calculator.solve()

if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, vect_test.sym.dot(vect_test.sym))
    pvmesh.point_data["Omega"] = vis.scalar_fn_to_pv_points(pvmesh, omega.sym)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, vect_test.sym)
    pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(pvmesh, vect_anal.sym)

    v_points = vis.meshVariable_to_pv_cloud(vect_test)
    v_points.point_data["V"] = vis.vector_fn_to_pv_points(v_points, vect_test.sym)
    v_points.point_data["V0"] = vis.vector_fn_to_pv_points(v_points, vect_anal.sym)
     
    pl = pv.Plotter(window_size=(1000, 750))

    pl.add_mesh(
        pvmesh,
        cmap="Greys_r",
        edge_color="Black",
        show_edges=False,
        scalars="Omega",
        use_transparency=False,
        opacity=0.3,
        # clim=[0,0.1],
    )

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    pl.add_arrows( v_points.points, 
                   v_points.point_data["V"], 
                   mag=1, 
                  color="Green", 
                  show_scalar_bar=True)

    pl.add_arrows( v_points.points, 
                   v_points.point_data["V0"], 
                   mag=1, 
                  color="Blue", 
                  show_scalar_bar=True)

    
    pl.show()
# -





