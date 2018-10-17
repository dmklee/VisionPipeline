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

class basicCurve():
	def __init__(self,seed,edges,gradients,side=1):
		self.seed = tuple(seed)
		self.edges = edges
		self.gradients = gradients
		self.curv_avg = 0.0
		self.num_pts = 0
		self.tilt = gradients[self.seed]%(2*np.pi)
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

		self.children = []

		self.lAOI_err = 0.0
		self.rAOI_err = 0.0

	def updateArchive(self):	
		self.cArchive[2] = self.cArchive[1]
		self.cArchive[1] = self.cArchive[0]
		self.cArchive[0] = self.curv_avg
		self.tArchive[2] = self.tArchive[1]
		self.tArchive[1] = self.tArchive[0]
		self.tArchive[0] = self.tilt
		self.rArchive[2,:] = self.rArchive[1,:]
		self.rArchive[1,:] = self.rArchive[0,:]
		self.rArchive[0,:] = self.rtail
		self.lArchive[2,:] = self.lArchive[1,:]
		self.lArchive[1,:] = self.lArchive[0,:]
		self.lArchive[0,:] = self.ltail

	def revertToArchive(self):
		age = self.cArchive.size-1
		self.rtail = tuple(self.rArchive[age,:])
		self.ltail = tuple(self.lArchive[age,:])
		self.curv_avg = self.cArchive[age]
		self.tilt = self.tArchive[age]

	def expand(self):
		self.updateArchive()
		if self.num_pts == 20:
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
		if (self.num_pts/20)**0.5 % 1 == 0:
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
		if pt is None:
			realTail = self.rtail if side==1 else self.ltail
		else:
			realTail = pt
		modelTail, modelGrad = self.getModeledTail(side,pt)
		vec_err = np.subtract(realTail,modelTail)
		d_err = np.linalg.norm(vec_err)
		if side == 1:
			self.rDir = modelGrad-side*np.pi/2
		else:
			self.lDir = modelGrad-side*np.pi/2
		th_err = abs(angleDiff(modelGrad,self.gradients[realTail]))
		return d_err, th_err

	def getCnewSingle(self,side,new_pt):
		new_tilt = self.gradients[new_pt]
		vec_StoNew = np.subtract(new_pt,self.seed)
		q_StoNew = np.linalg.norm(vec_StoNew)

		uvec_tilt = -np.array((np.sin(self.tilt),np.cos(self.tilt)))
		alpha = np.arccos(np.dot(vec_StoNew,uvec_tilt)/q_StoNew)
		theta = 2*(np.pi-alpha)
		c_loc = -np.sin(theta)/(q_StoNew*np.sin(alpha))
		# c_loc = 2*np.dot(vec_StoNew,uvec_tilt)/(q_StoNew)**2

		# th_diff = angleDiff(new_tilt,self.tilt)
		# c_grad = self.side*2*np.sin(th_diff/2.)/q_StoNew

		# self.c_grad = c_grad
		# self.c_loc = c_loc
		# z = np.clip((self.num_pts)/20,0,1.0)
		# c_new = z*c_loc+(1-z)*c_grad
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
		alpha = 0.3# if self.num_pts <= 10 else 1/self.num_pts
		self.curv_avg += alpha*(c_new-self.curv_avg)
		# self.curv_avg = -1/150.
	
	def growTail(self,side,pt):
		direction = self.gradients[pt]-side*np.pi/2.
		# if self.num_pts > 20:
		# 	if side == 1:
		# 		direction -= self.rAOI_err
		# 	else:
		# 		direction -= self.lAOI_err
		AOI_id = basicCurve.selectAOI(direction)
		AOI = self.AOIs[AOI_id]
		new_pt = self.sampleAOI(AOI,pt)
		vec = np.subtract(new_pt,pt)
		err = angleDiff(np.arctan2(vec[1],vec[0]),direction)
		print(err)
		if side==1:
			self.rAOI_err = err
		else:
			self.lAOI_err = err
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

	def sampleAOI(self,AOI,center):
		points = AOI+np.array(center)
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

if __name__ == "__main__":
	# name = "simple_shapes.png"
	name = 'occlusion2.png'
	# name = 'linking_testbed.png'
	img = testImage(mode='gaussian',v=0.0,name=name)
	img = gaussianFilter(img)
	edges,gradients = sobelOp(img)
	plt.figure(figsize=(10,6))
	plt.imshow(edges,cmap='gray')
	plt.autoscale(False)
	plt.tight_layout()

	# seeds = edgeSniffer(edges,grouping=400,style='absolute')
	seeds = [(130,152)]
	seeds = [seeds[0]]

	paths = []
	confs = []
	growth_steps = 400
	curv_data = np.empty((growth_steps))
	path_data, = plt.plot([],[],'b-',linewidth=2.5)
	anchor_data, = plt.plot([],[],'r^',markersize=6)
	colorscale = plt.get_cmap('Reds')

	seed_id = 0
	while True:
		if seed_id > 0:
			break
		try: 
			seed = seeds[seed_id]
		except IndexError:
			break
		seed_id += 1

		curve = basicCurve(seed,edges,gradients)
		plt.plot(curve.seed[1],curve.seed[0],'r.',markersize=5)
		
		DistErr = np.zeros((growth_steps,2))
		ThErr = np.zeros((growth_steps,2))
		Conf = np.zeros((growth_steps))
		
		for i in xrange(growth_steps):
			if curve.status == 'dead':
				seeds.append(curve.rtail)
				# seeds.append(curve.ltail)
				break
			curve.expand()

			# curv_data[i] = curve.curv_avg
			plt.plot(curve.rtail[1],curve.rtail[0],'g.',markersize=3.5)
			plt.plot(curve.ltail[1],curve.ltail[0],'g.',markersize=3.5)
			# DistErr[i] = curve.getGrowthErr(1)[0],curve.getGrowthErr(-1)[0]
			# ThErr[i] = curve.getGrowthErr(1)[1],curve.getGrowthErr(-1)[1]
			# Conf[i] = curve.conf
			# rModelTail = curve.getModeledTail(1)
			# lModelTail = curve.getModeledTail(-1)
			# plt.plot(rModelTail[1],rModelTail[0],'r.',markersize=2.5)
			# plt.plot(lModelTail[1],lModelTail[0],'r.',markersize=2.5)
			if i % 1 == 0:
				path = curve.path()
				path_data.set_data(path[:,1],path[:,0])
				path_data.set_color(colorscale(1-8*curve.conf/np.pi))
				anchors = np.vstack((curve.lAnchor,curve.rAnchor))
				anchor_data.set_data(anchors[:,1],anchors[:,0])
				plt.draw()
				plt.pause(0.1)

		path = curve.path()
		path_data.set_data(path[:,1],path[:,0])
		path_data.set_color(colorscale(1-8*curve.conf/np.pi))
		plt.draw()
		plt.pause(0.001)

		paths.append(path)
		confs.append(curve.conf)

		# plt.figure(figsize=(10,8))
		# plt.plot(curv_data,'g-')
		# plt.title('curvature during growth')

		# f, ax = plt.subplots(2,figsize=(10,8))
		# ax[0].plot(DistErr[:i,0],'r-',DistErr[:i,1],'b-')
		# ax[0].legend(('Right Side','Left Side'))
		# ax[0].set_title('Tail Distance Error over Time')

		# ax[1].plot(ThErr[:i,0],'r-',ThErr[:i,1],'b-')
		# ax[1].legend(('Right Side','Left Side'))
		# ax[1].plot(np.full(i,np.pi/8),'k:')
		# ax[1].plot(Conf[:i],'g-')
		# ax[1].set_title('Tail Gradient Error over Time')


	for i,path in enumerate(paths):
		if path.size > 1:
			c = 1-8*confs[i]/np.pi
			plt.plot(path[:,1],path[:,0],'-',color=colorscale(c),linewidth=1.5)
	plt.draw()
	plt.show()

# adjust distance error thresholds based on curvature, a bigger circle can accept larger error

# keep two anchors on tab

# allow broken edges to spawn new curve seeds, see if we can span a circle, ellipse with one seed

# allow nearby seeds to conquer each other or join existing curves

# create a system for linking touching curves

# better sampling approach

# allow for seeds to move

