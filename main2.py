# Author: David Klee
# Date  : 9/2/18
#

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

def getIMG(name="circle.png"):
	filename = os.path.join(os.getcwd(), "Pics/"+name)
	img = mpimg.imread(filename)
	return img

def testImage(mode='none',m=0.0,v=0.01,d=0.05,name='circle.png'):
	# mode = "gaussian" => specify mean (m) and var (v)
	# mode = "s&p" => specify density (d)
	# mode = "speckle" => specify variance (v)
	img = getIMG(name=name)[:,:,0]
	if mode == 'gaussian' or mode == 'speckle':
		noisy = util.random_noise(img,mode=mode,mean=m,var= v)
	elif mode == "s&p":
		noisy = util.random_noise(img,mode=mode,amount=d)
	else:
		noisy = img
	return noisy

def sobelOp(img):
	Kx = np.array([[1.0,0,-1.0], [2.0,0.0,-2.0], [1.0,0.0,-1.0]])
	Ky = np.array([[1.0,2.0,1.0], [0.0,0.0,0.0], [-1.0,-2.0,-1.0]])
	Gx = ndimage.convolve(img,Kx,mode='nearest')
	Gy = ndimage.convolve(img,Ky,mode='nearest')
	edges = np.sqrt(np.multiply(Gx,Gx)+np.multiply(Gy,Gy))
	edges = edges/np.max(edges)
	gradients = np.arctan2(Gy,Gx)
	return edges, gradients

def gaussianFilter(img,sigma=1.0):
	return ndimage.filters.gaussian_filter(img,sigma)

def angleDiff(new,old):
	# new minus old
	new = new%(2*np.pi)
	old = old%(2*np.pi)
	diff = (new-old)
	if diff > np.pi:
		diff = diff-2*np.pi
	if diff < -np.pi:
		diff = 2*np.pi+diff
	return diff

def curveDetector():
	img = testImage(mode='gaussian',v=0.00,name='circle.png')
	img = gaussianFilter(img,sigma=1.0)
	edges,gradients = sobelOp(img)
	seeds = findSeeds(edges)
	curves = []
	for seed in seeds:
		new_curve = basicCurve(seed,edges,gradients)
		if new_curve.findRooting():
			curves.append(new_curve)
	step_num = 0
	while True:
		step_num += 1
		still_alive = False
		for curve in curves:
			if curve.status == 'growing':
				still_alive = True
				curve.grow()
		if not still_alive or step_num > 20:
			break

def edgeSniffer(edges,grouping=40,style='absolute'):
	#take an image of edge likelihoods
	# finds the best edge candidate in a section of grouping^2 pixels
	# returns a list of indices
	w,h = edges.shape
	dividend = h//grouping
	remainder = h%grouping
	g = grouping
	trimmed = edges[:w-w%g,:h-h%g]
	tw,th = trimmed.shape
	sections =  np.array(np.hsplit(
					np.concatenate(
						np.split(trimmed,w//g),
							axis=1),(w//g)*(h//g)))
	a,b,c = sections.shape
	sections = sections.reshape(a,b*c)
	if style == 'absolute':
		maxes = np.amax(sections,axis=1)
		argmaxes = np.argmax(sections,axis=1)
	elif style == 'relative':
		maxes = np.empty(a)
		argmaxes = np.empty(a,dtype=int)
		ind_range = np.arange(b*c,dtype=int)
		for i,s in enumerate(sections):
			sum_val = np.sum(s)
			if sum_val > 0.0:
				argmaxes[i] = np.random.choice(ind_range,size=1,p=s/sum_val)
			else:
				argmaxes[i] = 0
			maxes[i] = s[argmaxes[i]]

	offsets = np.arange(argmaxes.size)
	indices = np.empty((a,2),dtype=int)
	indices[:,0] = g*(offsets[:]//(h//g)) + argmaxes[:]//g
	indices[:,1] = g*(offsets[:]%(h//g)) + argmaxes[:]%g

	#could use thresholding in the future
	indices = indices[np.where(maxes > 0.05)]

	return indices

class basicCurve():
	def __init__(self,seed,edges,gradients,side=1):
		self.seed = tuple(seed)
		self.edges = edges
		self.gradients = gradients
		self.curv_avg = 0.0
		self.curv_var = 0.0
		self.num_pts = 0
		self.tilt = gradients[seed[0],seed[1]]
		self.status = 'seeded'
		self.side = side
		self.AOIs = basicCurve.getAOIs()
		self.end = self.seed

	def getDirection(self,pt):
		return self.gradients[pt]-self.side*np.pi/2

	def getCnew(self,new_pt):
		new_tilt = self.gradients[new_pt]
		vec_StoNew = np.subtract(new_pt,self.seed)
		q_StoNew = np.linalg.norm(vec_StoNew)
		th_diff = angleDiff(new_tilt,self.tilt)
		c_grad = self.side*2*np.sin(th_diff/2.)/q_StoNew

		return c_grad

	def updateCurvature(self,new_pt):
		c_new = self.getCnew(new_pt)
		old_curv_avg = self.curv_avg
		self.curv_avg += (c_new-self.curv_avg)/self.num_pts
		self.curv_var += (c_new-old_curv_avg)*(c_new-self.curv_avg)

	def grow(self):
		direction = self.getDirection(self.end)
		AOI_id = basicCurve.selectAOI(direction)
		AOI = self.AOIs[AOI_id]
		new_pt = self.sampleAOI(AOI)
		self.num_pts += 1
		self.updateCurvature(new_pt)
		self.end = new_pt

	def path(self,res=5):
		# res is the number of pixels covered by a path point
		vec_StoEnd = np.subtract(self.end,self.seed)
		q_StoEnd = np.linalg.norm(vec_StoEnd)
		direction = self.tilt-self.side*np.pi/2.
		if abs(self.curv_avg) < 0.00001:
			return np.subtract(self.seed,q_StoEnd*np.array(((0,0),(np.sin(direction),np.cos(direction)))))
		radius = abs(1/self.curv_avg)
		th_prog = 2*np.arcsin(np.clip(q_StoEnd*self.curv_avg/2.,-1,1))
		th_inc = np.linspace(0.0,th_prog,num=self.num_pts//res)
		vec = np.empty((self.num_pts//res,2))
		vec[:,0] = radius*(1-np.cos(th_inc))
		vec[:,1] = radius*np.sin(th_inc)
		R = np.mat(((1,0),(0,1)))
		R = np.matrix(((np.cos(direction),-np.sin(direction)),
					  (np.sin(direction),np.cos(direction))))
		vec = np.mat(vec)*R
		return np.subtract(self.seed,np.sign(self.curv_avg)*vec)

	def sampleAOI(self,AOI):
		points = AOI+np.array(self.end)
		best_val = 0
		best_pt = (0,0)
		for pt in points:
			pt = tuple(pt)
			try:
				val = self.edges[pt]
				if val > best_val:
					best_val = val
					best_pt = pt
			except IndexError:
				pass
		return best_pt

	def findRooting(self):
		angles = np.zeros(25)
		weights = np.zeros(25)
		for i in range(-2,3):
			for j in range(-2,3):
				try:
					angles[5*i+j] = self.gradients[seed[0]+i,seed[1]+j]
					weights[5*i+j] = self.edges[seed[0]+i,seed[1]+j]
				except IndexError:
					pass
		vec = np.array((np.dot(weights,np.cos(angles)),np.dot(weights,np.sin(angles))))
		self.tilt = np.arctan2(vec[1],vec[0])
		if np.linalg.norm(vec) > 0.7*np.sum(weights) and \
				np.linalg.norm(vec) > 1.0:
			return True
		return False

	@staticmethod
	def getAOIs():
		AOIs = np.empty((8,3,2),dtype=int)
		#move in CCW direction
		AOIs[7] = np.array(((1,-1),(1,0),(1,1)))
		AOIs[6] = np.array(((0,1),(1,0),(1,1)))
		AOIs[5] = np.array(((1,1),(0,1),(-1,1)))
		AOIs[4] = np.array(((-1,1),(0,1),(-1,0)))
		AOIs[3] = np.array(((-1,-1),(-1,0),(-1,1)))
		AOIs[2] = np.array(((-1,0),(-1,-1),(0,-1)))
		AOIs[1] = np.array(((1,-1),(0,-1),(-1,-1)))
		AOIs[0] = np.array(((0,-1),(1,-1),(1,0)))
		return AOIs

	@staticmethod
	def selectAOI(angle):
		#returns the index corresponding to the right AOI
		# direction is an angle
		angle = angle%(2*np.pi)
		#now angle is greater than 0
		index = (angle + np.pi/8)//(np.pi/4)
		return int(index+1)%8

if __name__ == "__main__":
	colors = plt.get_cmap('RdYlBu')

	img = testImage(name='circle.png')
	img = gaussianFilter(img)
	edges,gradients = sobelOp(img)
	plt.figure(figsize=(10,8))
	plt.imshow(edges,cmap='gray')
	start = time.time()
	seeds = [(54,449)]
	seeds = edgeSniffer(edges,grouping=400)
	for seed in seeds:
		curve = basicCurve(seed,edges,gradients)
		if not curve.findRooting():
			pass
		# plt.plot(curve.seed[1],curve.seed[0],'r*')
		plt.plot([curve.seed[1],curve.seed[1]+3*np.cos(curve.tilt)],
				 [curve.seed[0],curve.seed[0]+3*np.sin(curve.tilt)],'b-')
		for i in xrange(520):
			curve.grow()
			color = 'g.'
			# if curve.curv_avg < 0:
			# 	color = 'r.'
			# std_dev = (curve.curv_var/(curve.num_pts+1))**0.5
			# val = 1-min(1,std_dev/0.04)
			# plt.plot(curve.end[1],curve.end[0],'s',color=colors(val))
		path = curve.path()
		std_dev = (curve.curv_var/(curve.num_pts+1))**0.5
		val = 1-min(1,std_dev/0.02)
		plt.plot(path[:,1],path[:,0],'r-')	

	plt.show()



