import os
import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc
from skimage import feature
from skimage import measure
from skimage import util
import time 
import sys

def getIMG(name="occlusion2.png"):
	filename = os.path.join(os.getcwd(), "Pics/"+name)
	img = mpimg.imread(filename)
	return img[:,:,:3] # remove alpha channel

def convertToGrayscale(img):
	return np.max(img, axis=2)

def getThreeChannels(img):
	# r, g, b
	bw = np.mean(img, axis=2)
	rg = (img[:,:,0] - img[:,:,1])/2
	yb = (np.mean(img[:,:,:2],axis=2) - img[:,:,2])/2
	return bw, rg, yb

def gaussianFilter(img,sigma=1.0):
	return ndimage.filters.gaussian_filter(img,sigma)

def sobelOp(img):
	Kx = np.array([[1.0,0,-1.0], [2.0,0.0,-2.0], [1.0,0.0,-1.0]])
	Ky = np.array([[1.0,2.0,1.0], [0.0,0.0,0.0], [-1.0,-2.0,-1.0]])
	Gx = ndimage.convolve(img,Kx,mode='nearest')
	Gy = ndimage.convolve(img,Ky,mode='nearest')
	E = np.abs(Gx)+np.abs(Gy)
	return E, Gx, Gy

def getGradID(pt, Gx, Gy):
	gx = Gx[tuple(pt)]
	gy = Gy[tuple(pt)]
	ratio = gx/gy if gy != 0 else gx/0.0001
	if abs(ratio) < 0.5:
		return 0
	if (ratio > 0.5 and ratio < 2):
		return 1
	if (ratio > 2 or ratio < -2):
		return 2
	return 3


def moveAlongContour(E, Gx, Gy, pt):
	pts_explored = []
	d = np.array([(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1)])
	last_grad_id = getGradID(pt, Gx, Gy)
	for i in range(180):
		pts_explored.append(pt)
		grad_id = getGradID(pt, Gx, Gy)
		if (min(abs(last_grad_id-grad_id), abs(8+last_grad_id-grad_id)) > 2):
			grad_id += 4
		last_grad_id = grad_id
		max_val = E[tuple(pt + d[grad_id])]
		max_id = grad_id
		if max_val < E[tuple(pt + d[(grad_id+1) % 8])]:
			max_id = (grad_id+1) % 8
		elif max_val < E[tuple(pt + d[(grad_id+7) % 8])]:
			max_id = (grad_id+7) % 8
		pt = pt + d[max_id]
	return np.array(pts_explored)

img = getIMG("circle.png")
bw,rg,yb = getThreeChannels(img)
bw = gaussianFilter(bw)
ur = (370,64)
w = 60
sample = bw#[ur[0]:ur[0]+w,ur[1]:ur[1]+w]
E, Gx, Gy = sobelOp(sample)
G_ids = np.zeros_like(Gx)
for i in range(G_ids.shape[0]):
	for j in range(G_ids.shape[1]):
		G_ids[i,j] = getGradID((i,j),Gx,Gy)

plt.figure(figsize=(8,6))
plt.imshow(G_ids,cmap='jet')
plt.colorbar()

plt.autoscale(False)
plt.axis('equal')
plt.tight_layout()
pts = moveAlongContour(E,Gx,Gy, (169,145))
curvature = np.zeros(len(pts))
for i in range(1,len(pts)):
	g1 = getGradID(pts[i-1],Gx, Gy)
	print(g1)
	g2 = getGradID(pts[i],Gx, Gy)
	diff = abs(g2 - g1)
	if diff > 4: diff = 8 - diff
	curvature[i] = g2
curvature = np.convolve(curvature, np.ones(2), mode='same')
for i, pt in enumerate(pts):
	plt.plot(pt[1],pt[0],'r.')
	# plt.arrow(pt[1], pt[0], Gy[tuple(pt)], -Gx[tuple(pt)],color='r')
# plt.xlim((0,w))
# plt.ylim((w,0))
plt.figure()
plt.plot(curvature, '.')
plt.show()
