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

def get_vec(pt,angle,l=2):
	vec = np.empty((2,2))
	vec[:,0] = pt[0],pt[0]+l*np.sin(angle)
	vec[:,1] = pt[1],pt[1]+l*np.cos(angle)
	return vec

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

def edgeSniffer(edges,grouping=30,style='absolute'):
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
	indices = indices[np.where(maxes > 0.1)]

	return indices

class basicCurve(object):
	def __init__(self,seed,edges,gradients,curve_map,curve_id):
		self.curve_id = curve_id
		self.curve_map = curve_map
		self.edges = edges
		self.gradients = gradients
		self.seed = tuple(seed)
		self.curv_avg = 0.0
		self.num_pts = 0
		self.tilt = self.gradients[self.seed]%(2*np.pi)
		self.status = 'dual' #'right','left','dead'
		self.AOIs = basicCurve.getAOIs(0)
		self.rtail = self.seed
		self.ltail = self.seed
		self.rArchive = np.zeros((3,2),dtype=int)
		self.lArchive = np.zeros((3,2),dtype=int)
		self.cArchive = np.zeros(3)
		self.tArchive = np.zeros(3)
		#right is 1, left is -1
		self.lAnchor = tuple(seed)
		self.rAnchor = tuple(seed)
		self.conf = 0.0

		self.rchild = None
		self.lchild = None

	def reseed(self,seed):
		best_val = 0
		for i in np.arange(-2,3):
			for j in np.arange(-2,3):
				px,py = i+seed[0],j+seed[1]
				try:
					if self.edges[px,py] > best_val:
						best_val = self.edges[px,py]
						seed = px,py
				except IndexError:
					pass
		return seed

	def updateArchive(self):	
		self.curve_map[tuple(self.rArchive[2,:])] = self.curve_id
		self.curve_map[tuple(self.lArchive[2,:])] = self.curve_id
		
		self.cArchive[2] 	= self.cArchive[1]
		self.cArchive[1] 	= self.cArchive[0]
		self.cArchive[0] 	= self.curv_avg
		self.tArchive[2] 	= self.tArchive[1]
		self.tArchive[1] 	= self.tArchive[0]
		self.tArchive[0] 	= self.tilt
		self.rArchive[2,:] 	= self.rArchive[1,:]
		self.rArchive[1,:] 	= self.rArchive[0,:]
		self.rArchive[0,:] 	= self.rtail
		self.lArchive[2,:] 	= self.lArchive[1,:]
		self.lArchive[1,:] 	= self.lArchive[0,:]
		self.lArchive[0,:] 	= self.ltail

	def revertToArchive(self):
		age = self.cArchive.size-1
		self.rtail = tuple(self.rArchive[age,:])
		self.ltail = tuple(self.lArchive[age,:])
		self.curv_avg = self.cArchive[age]
		self.tilt = self.tArchive[age]

	def propagate(self):
		while self.status != 'dead':
			self.expand()
			if self.num_pts > 400:
				self.status = 'dead'
		if self.checkIfValid():
			self.rchild = extCurve(self.rtail,self.edges,self.gradients,
						self.curve_map,self.curve_id,1)
			self.lchild = extCurve(self.ltail,self.edges,self.gradients,
						self.curve_map,self.curve_id,-1)
			rgrowing = True
			lgrowing = True
			while rgrowing or lgrowing:
				if rgrowing:
					rgrowing = self.rchild.propagate()
				if lgrowing:
					lgrowing = self.lchild.propagate()
		self.determineShape()

	def expand(self):
		self.updateArchive()
		if self.num_pts == 15:
			self.AOIs = basicCurve.getAOIs(spacing=1)
		rd_err,rth_err = self.getGrowthErr(1)
		ld_err,lth_err = self.getGrowthErr(-1)
		if self.num_pts > 10:
			rAnchor_err = self.getGrowthErr(1,self.rAnchor)[0]
			lAnchor_err = self.getGrowthErr(-1,self.lAnchor)[0]
		else:
			rAnchor_err,lAnchor_err = 0.0,0.0
		if self.num_pts > 0:
			th_err = rth_err+lth_err
			if self.status=='left':
				th_err -= rth_err
			if self.status=='right':
				th_err -= th_err
			self.conf += (th_err-self.conf)/self.num_pts
		if (self.num_pts/15)**0.5 % 1 == 0:
			self.setNewAnchor(1)
			self.setNewAnchor(-1)
		if self.status == 'dual':
			self.expandDual()
			if max(rd_err,2*rAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'left'
			if max(ld_err,2*lAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'right'
		elif self.status =='right':
			self.expandSingle(1)			
			if max(rd_err,ld_err,2*rAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'
		elif self.status == 'left':
			self.expandSingle(-1)
			if max(rd_err,ld_err,2*lAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'

	def setNewAnchor(self,side):
		age = self.cArchive.size-1
		if side == 1:
			self.rAnchor = tuple(self.rArchive[age,:])
		else:
			self.lAnchor = tuple(self.lArchive[age,:])

	def getGrowthErr(self,side,pt=None):
		isAnchor = False
		if pt is None:
			realTail = self.rtail if side==1 else self.ltail
		else:
			isAnchor = True
			realTail = pt
		modelTail, modelGrad = self.getModeledTail(side,pt)
		vec_err = np.subtract(realTail,modelTail)
		d_err = np.linalg.norm(vec_err)
		if side == 1 and not isAnchor:
			self.rDir = modelGrad-side*np.pi/2
			self.rModelTail = (0.5+modelTail).astype(int)
		elif side == -1 and not isAnchor:
			self.lDir = modelGrad-side*np.pi/2
			self.lModelTail = (0.5+modelTail).astype(int)
		th_err = abs(angleDiff(modelGrad,self.gradients[realTail]))
		if th_err > np.pi/2:
			th_err = abs(np.pi-th_err)
		return d_err, th_err

	def getCnewSingle(self,side,new_pt):
		new_tilt = self.gradients[new_pt]
		vec_StoNew = np.subtract(new_pt,self.seed)
		q_StoNew = np.linalg.norm(vec_StoNew)

		uvec_tilt = -np.array((np.sin(self.tilt),np.cos(self.tilt)))
		alpha = np.arccos(np.dot(vec_StoNew,uvec_tilt)/q_StoNew)
		theta = 2*(np.pi-alpha)
		c_loc = -np.sin(theta)/(q_StoNew*np.sin(alpha))
		return -side*c_loc

	def getCnewDual(self):
		vec_StoR = np.subtract(self.rtail,self.seed)
		q_StoR = np.linalg.norm(vec_StoR)
		vec_StoL = np.subtract(self.ltail,self.seed)
		q_StoL = np.linalg.norm(vec_StoL)
		vec_LtoR = np.subtract(self.rtail,self.ltail)
		q_LtoR = np.linalg.norm(vec_LtoR)
		sgn = 1 if np.cross(vec_StoL,vec_StoR) >= 0 else -1
		angle = np.arccos(np.clip(np.dot(vec_StoR,vec_StoL)
					/(q_StoL*q_StoR),-1,1))
		c_loc = -sgn*2*np.sin(angle)/q_LtoR

		th_ltail = self.gradients[self.ltail]
		th_rtail = self.gradients[self.rtail]
		th_diff = angleDiff(th_rtail,th_ltail)%(2.*np.pi)
		c_grad = -sgn*2*np.sin(th_diff/2.)/q_LtoR

		z = np.clip((self.num_pts)/20,0,1.0)
		c_new = z*c_loc+(1-z)*c_grad
		return c_new

	def updateCurv(self,side=0):
		if side == 1:
			c_new = self.getCnewSingle(side,self.rtail)
		elif side == -1:
			c_new = -self.getCnewSingle(side,self.ltail)
		else:
			c_new = self.getCnewDual()
		self.curv_avg += 0.3*(c_new-self.curv_avg)
	
	def growTail(self,side,pt):
		direction = self.gradients[pt]-side*np.pi/2.
		AOI_id = basicCurve.selectAOI(direction)
		AOI = self.AOIs[AOI_id]
		new_pt = self.sampleAOI(AOI,pt)
		modelDir = self.rDir if side == 1 else self.lDir
		if abs(angleDiff(modelDir+side*np.pi/2.,self.gradients[new_pt])) > np.pi/8:
			if side == 1:
				direction = self.rDir
				pt = tuple(self.rModelTail)
			else:
				direction = self.lDir
				pt = tuple(self.lModelTail)
			AOI_id = basicCurve.selectAOI(direction)
			AOI = self.AOIs[AOI_id]
			new_pt = self.sampleAOI(AOI,pt)

		old_pt = self.rArchive[2,:] if side==1 else self.lArchive[2,:]
		if self.edges[new_pt] < self.edges[tuple(old_pt)]/4:
			self.revertToArchive()
			if self.status == 'dual':
				self.status = 'right' if side == -1 else 'left'
			else:
				self.status = 'dead'
		self.num_pts += 1
		return new_pt

	def expandSingle(self,side):
		if side == 1:
			self.rtail = self.growTail(1,self.rtail)
		elif side == -1:
			self.ltail = self.growTail(-1,self.ltail)
		self.updateCurv(side)

	def expandDual(self):
		self.ltail = self.growTail(-1,self.ltail)
		self.rtail = self.growTail(1,self.rtail)
		self.updateTilt()
		self.updateCurv()

	def getTiltNew(self):
		vec_tilt = np.array((np.cos(self.tilt),np.sin(self.tilt)))
		dir_ltail = self.gradients[self.ltail]
		dir_rtail = self.gradients[self.rtail]
		l_vec = np.array((np.cos(dir_ltail),np.sin(dir_ltail)))
		r_vec = np.array((np.cos(dir_rtail),np.sin(dir_rtail)))
		rl_vec = np.add(r_vec,l_vec)
		tilt_grad = (np.arctan2(rl_vec[1],rl_vec[0]))
		if np.dot(rl_vec,vec_tilt) < 0:
			tilt_grad -= np.pi

		vSR = np.subtract(self.rtail,self.seed)
		qSR = np.linalg.norm(vSR)
		vSL = np.subtract(self.ltail,self.seed)
		qSL = np.linalg.norm(vSL)
		# we want to rotate vSR CCW 90 deg and vSL CW 90 deg
		vec_new = qSL*np.array((-vSR[1],vSR[0])) \
					+ qSR*np.array((vSL[1],-vSL[0]))
		sgn = np.sign(np.dot(vec_new,vec_tilt))
		
		tilt_loc = np.arctan2(vec_new[0],vec_new[1])
		if tilt_loc == 0.0:
			tilt_loc = self.tilt

		z = np.clip(((self.num_pts)/25),0,1.0) #z is the contribution of tilt_loc
		tilt_new = tilt_grad - z*angleDiff(tilt_grad,tilt_loc)

		# print(tilt_grad%(2*np.pi),tilt_loc%(2*np.pi),self.tilt)
		return tilt_new

	def updateTilt(self,mode='dual'):
		tilt_new = self.getTiltNew()

		delta = angleDiff(tilt_new,self.tilt)
		self.tilt += delta/self.num_pts
		self.tilt = self.tilt % (2*np.pi)

	def getModeledTail(self,side,pt=None):
		if side == 1:
			end = self.rtail
		else:
			end = self.ltail
		end = end if pt is None else pt
		vec_StoEnd = np.subtract(end,self.seed)
		q_StoEnd = np.linalg.norm(vec_StoEnd)
		th_prog = -side*2*np.arcsin(np.clip(q_StoEnd*self.curv_avg/2.,-1,1))
		direction = self.tilt+side*np.pi/2.
		if abs(self.curv_avg) != 0:
			radius = abs(1/self.curv_avg)
		else:
			return np.array(end),self.tilt
		vec = radius*np.array(((1-np.cos(th_prog)),np.sin(th_prog)))
		R = np.array(((np.cos(direction),np.sin(direction)),
						(-np.sin(direction),np.cos(direction))))
		vec = np.dot(R,vec)
		modeledTail = np.add(-side*np.sign(self.curv_avg)*vec,self.seed)
		modeledGrad = self.tilt+th_prog
		return modeledTail,modeledGrad

	def sidepath(self,side,res=8):
		# res is the number of pixels covered by a path point
		if side == 1:
			end = self.rtail
		else:
			end = self.ltail
		vec_StoEnd = np.subtract(end,self.seed)
		q_StoEnd = np.linalg.norm(vec_StoEnd)
		direction = self.tilt+side*np.pi/2.
		if abs(self.curv_avg) < 0.00001:
			return np.add(self.seed,q_StoEnd*np.array(((0,0),(np.sin(direction),np.cos(direction)))))
		radius = abs(1/self.curv_avg)
		th_prog = 2*np.arcsin(np.clip(q_StoEnd*self.curv_avg/2.,-1,1))
		th_inc = -side*np.linspace(0.0,th_prog,num=self.num_pts//res)
		vec = np.empty((self.num_pts//res,2))
		vec[:,0] = radius*(1-np.cos(th_inc))
		vec[:,1] = radius*np.sin(th_inc)
		R = np.mat(((np.cos(direction),-np.sin(direction)),
						(np.sin(direction),np.cos(direction)) ))
		vec = np.mat(vec)*R

		return np.add(self.seed,-side*np.sign(self.curv_avg)*vec)

	def path(self,res=5):
		lpath = self.sidepath(-1,res)
		rpath = self.sidepath(1,res)
		path = np.concatenate((lpath[::-1],rpath))
		return path

	def getAllPaths(self):
		paths = [self.path()]
		if self.rchild is not None:
			paths += self.rchild.getAllPaths()
		if self.lchild is not None:
			paths += self.lchild.getAllPaths()
		return paths

	def sampleAOI(self,AOI,center):
		points = AOI+np.array(center)
		best_val = -1
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

	def checkIfValid(self):
		if self.conf < np.pi/16 and self.num_pts > 15:
			return True
		return False

	def determineShape(self):
		print('i dont know what i am')

	@staticmethod
	def getAOIs(spacing=0):
		AOIs = np.empty((8,3,2),dtype=int)
		if spacing == 0:
			AOIs[7] = np.array(((1,-1),(1,0),(1,1))) # right side
			AOIs[6] = np.array(((0,1),(1,0),(1,1))) #top right
			AOIs[5] = np.array(((1,1),(0,1),(-1,1)))
			AOIs[4] = np.array(((-1,1),(0,1),(-1,0)))
			AOIs[3] = np.array(((-1,-1),(-1,0),(-1,1)))
			AOIs[2] = np.array(((-1,0),(-1,-1),(0,-1)))
			AOIs[1] = np.array(((1,-1),(0,-1),(-1,-1)))
			AOIs[0] = np.array(((0,-1),(1,-1),(1,0))) #bottom right

		#faster version
		if spacing == 1:
			AOIs[7] = np.array(((2,-1),(2,0),(2,1)))
			AOIs[6] = np.array(((1,1),(1,2),(2,1)))
			AOIs[5] = np.array(((1,2),(0,2),(-1,2)))
			AOIs[4] = np.array(((-1,1),(-1,2),(-2,1)))
			AOIs[3] = np.array(((-2,-1),(-2,0),(-2,1)))
			AOIs[2] = np.array(((-1,-1),(-1,-2),(-2,-1)))
			AOIs[1] = np.array(((1,-2),(0,-2),(-1,-2)))
			AOIs[0] = np.array(((1,-1),(1,-2),(2,-1)))
		return AOIs

	@staticmethod
	def selectAOI(angle):
		#returns the index corresponding to the right AOI
		# direction is an angle
		angle = angle%(2*np.pi)
		#now angle is greater than 0
		index = (angle + np.pi/8)//(np.pi/4)
		return int(index+1)%8

class extCurve(basicCurve):
	def __init__(self,seed,edges,gradients,curve_map,curve_id,side):
		super(extCurve,self).__init__(seed,edges,gradients,curve_map,curve_id)
		self.side = side
		self.child = None

	def propagate(self):
		if self.child is None:
			while self.status != 'dead':
				self.expand()
				if self.num_pts > 200:
					self.status = 'dead'
			if self.checkIfValid():
				new_seed = self.rtail if self.side == 1 else self.ltail
				self.child = extCurve(new_seed,self.edges,self.gradients,
							self.curve_map,self.curve_id,self.side)
				return True
			return False
		else:
			return self.child.propagate()

	def growTail(self,side,pt):
		new_pt = super(extCurve,self).growTail(side,pt)
		if self.num_pts > 10:
			if side == self.side:
				if self.curve_map[new_pt] == self.curve_id:
					self.status = 'dead'
		return new_pt

	def expand(self):
		self.updateArchive()
		if self.num_pts == 15:
			self.AOIs = basicCurve.getAOIs(spacing=1)
		rd_err,rth_err = self.getGrowthErr(1)
		ld_err,lth_err = self.getGrowthErr(-1)
		if self.num_pts > 10:
			rAnchor_err = self.getGrowthErr(1,self.rAnchor)[0]
			lAnchor_err = self.getGrowthErr(-1,self.lAnchor)[0]
		else:
			rAnchor_err,lAnchor_err = 0.0,0.0
		if self.num_pts > 0:
			th_err = rth_err+lth_err
			if self.status=='left':
				th_err -= rth_err
			if self.status=='right':
				th_err -= th_err
			self.conf += (th_err-self.conf)/self.num_pts
		if (self.num_pts/15)**0.5 % 1 == 0:
			self.setNewAnchor(1)
			self.setNewAnchor(-1)
		if self.status == 'dual':
			self.expandDual()
			if max(rd_err,2*rAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'
				if self.side == -1:
					self.status = 'left'
			if max(ld_err,2*lAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'
				if self.side == 1:
					self.status = 'right'
		elif self.status =='right':
			self.expandSingle(1)			
			if max(rd_err,ld_err,2*rAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'
		elif self.status == 'left':
			self.expandSingle(-1)
			if max(rd_err,ld_err,2*lAnchor_err) > 2:
				self.revertToArchive()
				self.status = 'dead'
			
	def getAllPaths(self):
		paths = []
		if self.checkIfValid():
			paths.append(self.path())
			if self.child is not None:
				paths.extend(self.child.getAllPaths())
		return paths

if __name__ == "__main__":
	name = "simple_shapes.png"
	name = 'occlusion.png'
	# name = 'ellipses.png'
	img = testImage(mode='gaussian',v=0.0,name=name)
	img = gaussianFilter(img)
	edges,gradients = sobelOp(img)
	curve_map = np.zeros(edges.shape)
	plt.figure(figsize=(10,6))
	plt.imshow(edges,cmap='gray')
	plt.autoscale(False)
	plt.tight_layout()

	seeds = edgeSniffer(edges,grouping=200,style='absolute')
	# seed = (130,140)

	paths = []

	seed_id = 1
	curve = basicCurve(seed,edges,gradients,curve_map,seed_id)
	curve.propagate()
	paths = curve.getAllPaths()

	for path in paths:
		plt.plot(path[:,1],path[:,0],'-',linewidth=3.5)
		plt.draw()
		plt.pause(0.01)
	plt.show()

# adjust distance error thresholds based on curvature, a bigger circle can accept larger error

# keep two anchors on tab

# allow broken edges to spawn new curve seeds, see if we can span a circle, ellipse with one seed

# allow nearby seeds to conquer each other or join existing curves

# create a system for linking touching curves

# better sampling approach

# allow for seeds to move

