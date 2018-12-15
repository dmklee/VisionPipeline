import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc
from skimage import feature
from skimage import measure
from skimage import util
import time

def getIMG(name="toyblocks.png"):
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

img = getIMG()
bw,rg,yb = getThreeChannels(img)
plt.figure(figsize=(8,6))
plt.imshow(yb,cmap='gray')
plt.autoscale(False)
plt.tight_layout()
plt.show()