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

import numpy as np
import sympy


# %%
import skimage as ski
import pyvista as pv


# +
blocks = ski.io.imread("FaultedBlocksImage.tiff" )
blocks_T = blocks.transpose(1,0,2)
width = blocks.shape[1]/blocks.shape[0]

from scipy import ndimage
from skimage.util import random_noise
from skimage import feature
from skimage.color import rgb2gray


# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,6))
ax = plt.subplot(1,1,1)
ax.imshow(blocks_T.transpose(1,0,2))

# Extract information from the legend (currently, that means "by hand")

materials = {
    "fault_R": (222,8,8),
    "fault_B1": (68,55,230),
    "fault_B2": (56, 75, 187),
    "source": (94, 228, 213),
    "basement": (145, 135, 135),
    "layer_1": (76, 155, 75),
    "upper_fw": (163, 224, 112),
    "upper_hw": (214, 185, 31),
  
}

def match_colour(name):

    range = 1

    # R = np.logical_and(blocks_F[:,:,0] > materials[name][0] - range, blocks_F[:,:,0] < materials[name][0] + range)
    # G = np.logical_and(blocks_F[:,:,1] > materials[name][1] - range, blocks_F[:,:,1] < materials[name][1] + range)
    # B = np.logical_and(blocks_F[:,:,2] > materials[name][2] - range, blocks_F[:,:,2] < materials[name][2] + range)

    R = np.abs(blocks_T[:,:,0] - materials[name][0]) < range
    G = np.abs(blocks_T[:,:,1] - materials[name][1]) < range
    B = np.abs(blocks_T[:,:,2] - materials[name][2]) < range
    
    mask = np.logical_and(R, np.logical_and(G,B))
    
    return (mask)

# make this a mask dictionary. I'm not sure 
# why the labeling misses so many pixels. A fuzzy match is also possible but the 
# best would probably be "nearest pixel"

label = np.zeros_like(blocks_T[:,:,0])
fuzzy_label = np.zeros_like(blocks_T[:,:,0], dtype=float)

mat_masks = {}
for i, mat in enumerate(materials.keys()):
    mask = match_colour(mat)
    fuzzy_mask = ski.filters.gaussian(mask, sigma=2, mode="nearest" )
    mat_masks[mat] = mask
    
    label[mask] = i
    fuzzy_label += fuzzy_mask * i

new_label = label.copy()
new_label[label==0] = fuzzy_label[label==0]


# These regions / materials are where we would like to concentrate the 
# mesh refinement. An alternative, with a generic image is to use this
# lines = sato(grayscale_blocks, [1,2,3,4], black_ridges=True)

lmask = mat_masks["fault_R"] | mat_masks["fault_B1"] | mat_masks["fault_B2"] | mat_masks["source"]
blurred_lines = ski.filters.gaussian(lmask, sigma=3, mode="nearest" )
blurred_lines = 50.0 * np.minimum(blurred_lines, 0.02)

# -

# %%
materials.keys()

# %%
# Segmentation / label

from skimage import data, segmentation, color
from skimage import graph
from matplotlib import colormaps
from matplotlib import pyplot as plt

cmap = colormaps['summer'].resampled(8)

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(12, 6))

ax.imshow(new_label.T, cmap=cmap)
ax.axis('off')

plt.tight_layout()

# +
# Segmentation / label

from skimage import data, segmentation, color
from skimage import graph
from matplotlib import colormaps
from matplotlib import pyplot as plt

img = blocks_T
labels1 = segmentation.slic(img, compactness=3, 
                            sigma=4,
                            n_segments=12, 
                            channel_axis=2, 
                            mask=(blurred_lines>0.95))


cmap = colormaps['jet'].resampled(24)

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(12, 8))

ax.imshow(blurred_lines.T, cmap=cmap)
ax.axis('off')

plt.tight_layout()

# %%
# +
mesh0 = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0,0.0),
            maxCoords=(width,1),
            cellSize=0.02
)

H = uw.discretisation.MeshVariable("H", mesh0, 1)
Metric = uw.discretisation.MeshVariable("M", mesh0, 1, degree=1)

# -

sample_coords = (0,1) + (1,-1) * H.coords
image_coords = sample_coords * ( blurred_lines.shape[0] / width,  blurred_lines.shape[1] )
mvals = ndimage.map_coordinates(blurred_lines**0.5, image_coords.T , order=1, mode='nearest').astype(np.float32)


with mesh0.access(H):
    H.data[:,0] = 10.0 + 40000 * mvals

# +
icoord, meshA = adaptivity.mesh_adapt_meshVar(mesh0, H, Metric, redistribute=True)
HA = uw.discretisation.MeshVariable("HA", meshA, 1)
cell_label = uw.discretisation.MeshVariable("L", meshA, 1, degree=0, continuous=False)

## Remap the information

sample_coords = (0,1) + (1,-1) * HA.coords
image_coords = sample_coords * ( blurred_lines.shape[0] / width,  blurred_lines.shape[1] )
mvals = ndimage.map_coordinates(blurred_lines, image_coords.T , order=1, mode='reflect').astype(np.float32)

with meshA.access(HA):
    HA.data[:,0] = 10.0 + 100000 * mvals

sample_coords = (0,1) + (1,-1) * cell_label.coords
image_coords = sample_coords * ( blurred_lines.shape[0] / width,  blurred_lines.shape[1] )
mvals = ndimage.map_coordinates(new_label, image_coords.T , order=0, mode='reflect')

with meshA.access(cell_label):
    cell_label.data[:,0] = mvals



# %%
if uw.mpi.size == 1:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh0 = vis.mesh_to_pv_mesh(mesh0)
    pvmeshA = vis.mesh_to_pv_mesh(meshA)
    pvmesh0.point_data["H"] = vis.scalar_fn_to_pv_points(pvmesh0, H.sym[0])
    pvmeshA.point_data["H"] = vis.scalar_fn_to_pv_points(pvmeshA, HA.sym[0])
    pvmeshA.cell_data["L"] = vis.scalar_fn_to_pv_points(pvmeshA.cell_centers(), cell_label.sym[0])

    pl = pv.Plotter(window_size=(1500, 750))

    pl.add_mesh(
                pvmeshA,
                cmap="rainbow",
                scalars="L",
                edge_color="Black",
                show_edges=True,
                use_transparency=False,
                opacity=0.8,
                show_scalar_bar=False,
                clim=[0,8]
               )
    
    pl.add_mesh(
                pvmeshA,
                style="wireframe",
                edge_color="Black",
                show_edges=True,
                use_transparency=False,
                opacity=0.5,
                show_scalar_bar=False,
                
               )


    pl.show(jupyter_backend='html')

# %%
# +
p_soln = uw.discretisation.MeshVariable("P", meshA, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", meshA, meshA.dim, degree=1, continuous=True)

darcy = uw.systems.SteadyStateDarcy(meshA, h_Field=p_soln, v_Field=v_soln)
darcy.constitutive_model = uw.constitutive_models.DarcyFlowModel
darcy.petsc_options.delValue("ksp_monitor")

# ['fault_R', 'fault_B1', 'fault_B2', 'source', 'basement', 'layer_1', 'upper_fw', 'upper_hw'])

darcy.constitutive_model.Parameters.permeability = sympy.Piecewise(
                                                    (10, sympy.Abs(cell_label.sym[0] - 0) < 0.1),   # fault_R
                                                    (0.01, sympy.Abs(cell_label.sym[0] - 1) < 0.1),  # fault_B1
                                                    (10.0, sympy.Abs(cell_label.sym[0] - 2) < 0.1),  # fault B2
                                                    (10.0, sympy.Abs(cell_label.sym[0] - 3) < 0.1),  # source region 
                                                    (0.5, sympy.Abs(cell_label.sym[0] - 4) < 0.1), 
                                                    (20.0, sympy.Abs(cell_label.sym[0] - 5) < 0.1), 
                                                    (0.01, sympy.Abs(cell_label.sym[0] - 6) < 0.1), 
                                                    (2.0, sympy.Abs(cell_label.sym[0] - 7) < 0.1), 
                                                    (10.0, True))


darcy.f = sympy.Piecewise((10.0, sympy.Abs(cell_label.sym[0] - 3) < 0.1), (0.0, True))
darcy.constitutive_model.Parameters.s = sympy.Matrix([0, -1]).T

darcy.tolerance = 1e-6

darcy.add_dirichlet_bc(0.0, "Top")

darcy._v_projector.smoothing = 1.0e-6

# -

darcy.petsc_options.setValue("snes_monitor", None)
darcy.solve(verbose=False)

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(meshA)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, v_soln.sym)
    pvmesh.point_data["P"] = vis.scalar_fn_to_pv_points(pvmesh, p_soln.sym)
    pvmesh.cell_data["L"] = vis.scalar_fn_to_pv_points(pvmesh.cell_centers(), cell_label.sym[0])

    velocity_points = vis.meshVariable_to_pv_cloud(v_soln)
    velocity_points.point_data["V"] = vis.vector_fn_to_pv_points(velocity_points, v_soln.sym)

    # point sources at cell centres

    skip=10
    points = np.zeros((meshA._centroids[::skip].shape[0], 3))
    points[:, 0] = meshA._centroids[::skip, 0]
    points[:, 1] = meshA._centroids[::skip, 1]
    point_cloud = pv.PolyData(points[::3])

    pvstream = pvmesh.streamlines_from_source(
                                                point_cloud,
                                                vectors="V",
                                                integrator_type=45,
                                                integration_direction="both",
                                                max_steps=1000,
                                                max_time=0.5,
                                                surface_streamlines=True,
                                            )

    pl = pv.Plotter(window_size=(1600,800))

    pl.add_mesh(
                pvmesh,
                cmap="coolwarm",
                edge_color="Black",
                show_edges=False,
                scalars="P",
                use_transparency=False,
                opacity=0.75,
                show_scalar_bar=True,
            )


    # pl.add_arrows(velocity_points.points, velocity_points.point_data["V"], mag=0.01, opacity=0.5, show_scalar_bar=False)
    pl.add_mesh(pvstream, line_width=1.0, show_scalar_bar=False)


    pl.add_mesh(
                pvmesh,
                cmap="rainbow",
                scalars="L",
                edge_color="Black",
                show_edges=False,
                use_transparency=False,
                opacity=0.3,
                edge_opacity=0.1,
                show_scalar_bar=False,
                clim=[0,8]
               )

    
    pl.add_mesh(
                pvmesh,
                style="wireframe",
                edge_color="Black",
                show_edges=True,
                use_transparency=False,
                opacity=0.3,
                edge_opacity=0.1,
                show_scalar_bar=False,
                clim=[0,8]
               )

    
    pl.show(cpos="xy", jupyter_backend="html")




# %%

# %%
p_soln.clean_name

# %%