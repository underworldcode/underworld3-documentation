# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# +
#|  echo: false  # Hide in html version

# This is required to fix pyvista 
# (visualisation) crashes in interactive notebooks (including on binder)

# -

import underworld3 as uw
import numpy as np
import sympy

# +
mesh1 = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0, cellSize=0.1)
x, y = mesh1.X

# Continuous function
print(f"{uw.mpi.rank} - define continuous variables", flush=True)
s_fn = sympy.cos(5.0 * sympy.pi * x) * sympy.cos(5.0 * sympy.pi * y)
s_soln = uw.discretisation.MeshVariable("S", mesh1, 1, degree=1)

# second mesh variable
print(f"{uw.mpi.rank} - define 2nd variable", flush=True)
s_values = uw.discretisation.MeshVariable("S2", mesh1, 2, degree=1, continuous=True)

# Projection operation
print(f"{uw.mpi.rank} - build projections", flush=True)
scalar_projection = uw.systems.Projection(mesh1, s_soln, verbose=False)
print(f"{uw.mpi.rank} - build projections ... done", flush=True)

scalar_projection.uw_function = s_values.sym[0]
scalar_projection.smoothing = 1.0e-6


# S2 coordinates
with mesh1.access():
    print(f"{uw.mpi.rank} ", s_values.coords[0:3].reshape(-1), flush=True)

# Values on S2
# print(f"{uw.mpi.rank} - set values", flush=True)
with mesh1.access(s_values):
    print(f"{uw.mpi.rank} ", s_values.data[0:3].reshape(-1), flush=True)
    # s_values.data[:, :] = 1.0 # uw.function.evalf(sympy.sympify(1), s_values.coords)

# +
# Try to grab the information directly from the mesh dm

mesh1.update_lvec()

names, isets, dms = mesh1.dm.createFieldDecomposition()

fields = {}
for i, field_name in enumerate(names):
    fields[field_name] = (isets[i], dms[i])


# -

mesh1.update_lvec()
gvec = mesh1.dm.getGlobalVec()

# +
# fields["S2"][0].view()
# -

indexset, subdm = mesh1.dm.createSubDM(s_values.field_id)



# +
mesh1.update_lvec()
print(f"{uw.mpi.rank}: {gvec.array.shape}, {mesh1.lvec.array.shape}")

mesh1.dm.localToGlobal(mesh1.lvec, gvec, addv=False)
mesh1.dm.globalToLocal(gvec, mesh1.lvec, addv=False)

# Get subdm / subvector

indexset, subdm = mesh1.dm.createSubDM(s_values.field_id)

slvec_S2 = subdm.getLocalVec()
sgvec_S2 = subdm.getGlobalVec()
slvec_S2.set(2.0)

subdm.localToGlobal(slvec_S2, sgvec_S2)
subdm.globalToLocal(sgvec_S2, slvec_S2)

subvec_S2 = gvec.getSubVector(indexset)
sgvec_S2.copy(subvec_S2)
gvec.restoreSubVector(indexset, subvec_S2)

mesh1.dm.globalToLocal(gvec, mesh1.lvec, addv=False)
mesh1.dm.localToGlobal(mesh1.lvec, gvec, addv=False)

subdm.destroy()
indexset.destroy()

mesh1.dm.restoreGlobalVec(gvec)

# -





# +
scalar_projection.solve()

print(f"{uw.mpi.rank} - solve projection", flush=True)
# mesh1.dm.view()


print(f"Finalised")
# -




