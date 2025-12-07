#!/usr/bin/env python  
#-*- coding:utf-8 _*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

idx = 0
start_idx = 6
path = '/root/files/pdessl/data/large/pdebench/ns3d_pdb_M1_rand/test/data_{}.hdf5'.format(idx)
data = h5py.File(path, 'r')['data'][...,start_idx, 2]
def volume_rendering(data, step=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    # Get data dimensions
    nx, ny, nz = data.shape

    # Generate x, y, z coordinates
    x, y, z = np.mgrid[0:nx:step, 0:ny:step, 0:nz:step]

    # Iterate each point and draw semi-transparent points
    for i in range(0, nx, step):
        for j in range(0, ny, step):
            for k in range(0, nz, step):
                ax.scatter(i, j, k, color='blue', alpha=data[i, j, k])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

x, y, z = np.mgrid[0:1:30j, 0:1:30j, 0:1:30j]
import plotly.graph_objects as go
# Create Plotly figure object
fig = go.Figure(data=go.Isosurface(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=data.flatten(),
    isomin=0.2,
    isomax=0.8,
    caps=dict(x_show=False, y_show=False),
    showscale=False
))

# Update layout
# fig.update_layout()
fig.update_layout(
    scene=dict(
        xaxis_title='',  # clear X axis title
        yaxis_title='',  # clear Y axis title
        zaxis_title='',  # clear Z axis title
        xaxis_showticklabels=False,  # hide X tick labels
        yaxis_showticklabels=False,  # hide Y tick labels
        zaxis_showticklabels=False   # hide Z tick labels
    )
)
# fig.tight_layout()
# Show figure
fig.show()

# Save as static image
fig.write_image("pdb3d.png")


# volume_rendering(data)
# plot_iso_surface(data)
