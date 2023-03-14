from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import math
import random

fig = plt.figure()
ax = fig.gca(projection='3d')

 
 
# Create axis
axes = [5, 5, 5]
 
# Create Data
data = np.ones(axes, dtype=np.bool)
 
# Control Transparency
alpha = 0.9
 
# Control colour
colors = np.empty(axes + [4], dtype=np.float32)
 
colors[0] = [1, 0, 0, alpha]  # red
colors[1] = [0, 1, 0, alpha]  # green
colors[2] = [0, 0, 1, alpha]  # blue
colors[3] = [1, 1, 0, alpha]  # yellow
colors[4] = [1, 1, 1, alpha]  # grey
 
# Plot figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
 
# Voxels is used to customizations of
# the sizes, positions and colors.
ax.voxels(data, facecolors=colors, edgecolors='grey')

#Environment and Obstacles
class env3d:

#environment class is defined by obstacle vertices and boundaries
	def __init__(self, x,y,z,xmin,xmax,ymin,ymax,zmin,zmax):
		self.x = x
		self.y = y
		self.z = z
		self.xmin=xmin
		self.xmax=xmax
		self.ymin=ymin
		self.ymax=ymax
		self.zmin=zmin
		self.zmax=zmax
		
				
#draw the edges of a 3d cuboid			
	def cubedraw(self, obsx, obsy, obzl, obzh, k):
		x = obsx
		y = obsy
		zl = [obzl,obzl,obzl,obzl,obzl]
		zh = [obzh,obzh,obzh,obzh,obzh]
		
		ax.plot(x, y, zl, k)
		ax.plot(x,y,zh,k)
		for i in range (0,len(x)-1):
			obx = [x[i],x[i]]
			oby = [y[i],y[i]]
			obz = [zl[i],zh[i]]
			ax.plot(obx,oby,obz,k)
						

if __name__ == "__main__":
	x = [0, 1,  1, 0, 0, 1,  1, 0]
	y = [0, 0, 1, 1, 0, 0, 1, 1]
	z = [0, 0, 0, 0, 1, 1, 1, 1]
	ax




