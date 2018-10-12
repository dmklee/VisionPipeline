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
		self.num_pts = 0
		self.tilt = gradients[self.seed]%(2*np.pi)
		self.tilt_var = 0.1
		self.status = 'dual' #'right','left','dead'
		self.AOIs = basicCurve.getAOIs(0)
		self.rtail = self.seed
		self.ltail = self.seed

		#right is 1, left is -1

		self.c_grad = 0
		self.c_loc = 0

	def expand(self):
		if self.num_pts == 20:
			# self.status = 'left'
			self.AOIs = basicCurve.getAOIs(spacing=1)
		if self.status == 'dual':
			self.expandDual()
		elif self.status =='right':
			self.expandSingle(1)
		elif self.status == 'left':
			self.expandSingle(-1)

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
		old_curv_avg = self.curv_avg
		self.curv_avg += 0.3*(c_new-self.curv_avg)
		# self.curv_avg = -1/150.
	
	def growTail(self,side,pt):
		direction = self.gradients[pt]-side*np.pi/2.
		AOI_id = basicCurve.selectAOI(direction)
		AOI = self.AOIs[AOI_id]
		new_pt = self.sampleAOI(AOI,pt)
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

		vec_StoR = np.subtract(self.rtail,self.seed)
		q_StoR = np.linalg.norm(vec_StoR)
		vec_StoL = np.subtract(self.ltail,self.seed)
		q_StoL = np.linalg.norm(vec_StoL)
		vec_new = vec_StoR/q_StoR + vec_StoL/q_StoL
		sgn = np.sign(np.dot(vec_new,vec_tilt))
		tilt_loc = (np.arctan2(vec_new[0],vec_new[1])-sgn*np.pi)

		z = np.clip(((self.num_pts-10)/20),0,1.0) #z is the contribution of tilt_loc
		tilt_new = tilt_grad - z*angleDiff(tilt_grad,tilt_loc)

		# print(tilt_grad%(2*np.pi),tilt_loc%(2*np.pi),self.tilt)
		return tilt_new

	def updateTilt(self,mode='dual'):
		tilt_new = self.getTiltNew()

		delta = angleDiff(tilt_new,self.tilt)
		self.tilt += delta/self.num_pts
		self.tilt = self.tilt % (2*np.pi)

	def getModeledTail(self,side):
		if side == 1:
			end = self.rtail
		else:
			end = self.ltail
		vec_StoEnd = np.subtract(end,self.seed)
		q_StoEnd = np.linalg.norm(vec_StoEnd)
		th_prog = -side*2*np.arcsin(np.clip(q_StoEnd*self.curv_avg/2.,-1,1))
		direction = self.tilt+side*np.pi/2.
		try:
			radius = abs(1/self.curv_avg)
		except ZeroDivisionError:
			return np.array(end)
		vec = radius*np.array(((1-np.cos(th_prog)),np.sin(th_prog)))
		R = np.array(((np.cos(direction),np.sin(direction)),
						(-np.sin(direction),np.cos(direction))))
		vec = np.dot(R,vec)
		return np.add(-side*np.sign(self.curv_avg)*vec,self.seed)

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
			return np.subtract(self.seed,q_StoEnd*np.array(((0,0),(np.sin(direction),np.cos(direction)))))
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
	name = "circles.png"

	img = testImage(mode='gaussian',v=0.01,name=name)
	img = gaussianFilter(img,sigma=1)
	edges,gradients = sobelOp(img)
	plt.figure(figsize=(10,8))
	plt.imshow(edges,cmap='gray')
	start = time.time()
	seeds = edgeSniffer(edges,grouping=400)
	seeds = [(383,270)]
	# seeds = [seeds[0]]

	growth_steps = 82
	curv_data = np.empty((growth_steps,4))
	path_data, = plt.plot([],[],'b-',linewidth=2.5)
	for seed in seeds:
		curve = basicCurve(seed,edges,gradients)
		plt.plot(curve.seed[1],curve.seed[0],'r.',markersize=10)
	
		
		for i in xrange(growth_steps):
			curve.expand()
			if i > 0:
				delta_c = curve.curv_avg-curv_data[i-1,0]
			else:
				delta_c = 0
			curv_data[i] = (curve.curv_avg,curve.c_loc,curve.c_grad,delta_c)
			# plt.plot(curve.rtail[1],curve.rtail[0],'g.',markersize=2.5)
			# plt.plot(curve.ltail[1],curve.ltail[0],'g.',markersize=2.5)
			if i % 10 == 0:
				path = curve.path()
				path_data.set_data(path[:,1],path[:,0])
				plt.draw()
				plt.pause(0.0001)
		path = curve.path()
		path_data.set_data(path[:,1],path[:,0])
		# plt.figure()
		# plt.plot(curv_data[:,1],'r-',linewidth=1.5)
		# plt.plot(curv_data[:,2],'b-',linewidth=1.5)
		# plt.plot(curv_data[:,0],'g-')
		# plt.legend(('location-based','gradient-based','average'))
		# plt.title('Curvature Data during Growth')

		# if name == "small_circle.png":
		# 	real_curv = 1/150.
		# 	plt.plot(np.full(growth_steps,real_curv),'k--',linewidth=1.5)
		# 	plt.ylim((0.75*real_curv,1.25*real_curv))
		# if name == "circle.png":
		# 	real_curv = 1/185.
		# 	plt.plot(np.full(growth_steps,real_curv),'k--',linewidth=1.5)
		# 	plt.ylim((0.5*real_curv,1.5*real_curv))

	plt.draw()
	plt.pause(1)
	plt.show()




