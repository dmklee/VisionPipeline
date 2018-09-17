# Author: David Klee
# Date  : 9/2/18
#
#
#
import os
import numpy as np 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from skimage import measure

def getIMG(name="circle"):
	filename = os.path.join(os.getcwd(), "Pics/"+name + ".png")
	img = mpimg.imread(filename)
	return img

def showIMG(img):
	plt.figure()
	plt.imshow(img)
	plt.show()

def seeIMGbyEdges(img):
	f, axarr = plt.subplots(3,3,figsize=(7, 7))
	axarr[0][0].imshow(img)
	axarr[0][0].axis('off')
	for i in range(1,9):
		edges = getEdges(img[:,:,2],i-1)
		axarr[i//3][i%3].imshow(edges)
		axarr[i//3][i%3].axis('off')
	plt.tight_layout()
	plt.show()

class Image():
	def __init__(self,filename):
		self.name = filename
		fullname = os.path.join(os.getcwd(),"Pics/" + filename + ".png")
		self.image = mpimg.imread(fullname)[:,:,:3]
		self.dimensions = len(self.image),len(self.image[0])
		self.R = smoothImage(self.image[:,:,0]).flatten() #from 0-1
		self.G = smoothImage(self.image[:,:,1]).flatten() #from 0-1
		self.B = smoothImage(self.image[:,:,2]).flatten() #from 0-1
		self.RGB = np.vstack((self.R,self.G,self.B))
		self.makeHSV()
		self.makeHSL()
		self.makeBW()

	def makeHSV(self):
		Cmax = np.amax(self.RGB,axis=0)
		Cmin = np.amin(self.RGB,axis=0)
		delta = Cmax - Cmin

		def getH(d,cmax,r,g,b):
			#in degrees
			if d < 0.0001:
				h = 0
			elif cmax == r:
				h = 60*( ((g-b)/d) % 6 )
			elif cmax == g:
				h = 60*( (b-r)/d + 2 )
			elif cmax == b:
				h = 60*( (r-g)/d + 4 )
			return h/360.

		def getS(d,cmax):
		 	if cmax < 0.0001:
		 		return 0
		 	return d/cmax

		def getV(cmax):
			return cmax

		H = np.empty(len(self.R))
		S = np.empty(len(self.R))
		V = np.empty(len(self.R))
		for i in xrange(len(self.R)):
			H[i] = getH(delta[i],Cmax[i],self.R[i],self.G[i],self.B[i])
			S[i] = getS(delta[i],Cmax[i])
			V[i] = getV(Cmax[i])
		self.HSV = np.vstack((H,S,V))

	def makeHSL(self):
		Cmax = np.amax(self.RGB,axis=0)
		Cmin = np.amin(self.RGB,axis=0)
		delta = Cmax - Cmin

		def getH(d,cmax,r,g,b):
			#in degrees
			if d < 0.0001:
				h = 0
			elif cmax == r:
				h = 60*( ((g-b)/d) % 6 )
			elif cmax == g:
				h = 60*( (b-r)/d + 2 )
			elif cmax == b:
				h = 60*( (r-g)/d + 4 )
			return h/360.

		def getS(d,cmax,cmin):
		 	if abs(d) < 0.0001:
		 		return 0
		 	return d/(1-abs(cmax+cmin-1))

		def getL(cmax,cmin):
			return (cmax+cmin)/2.0

		H = np.empty(len(self.R))
		S = np.empty(len(self.R))
		L = np.empty(len(self.R))
		for i in xrange(len(self.R)):
			H[i] = getH(delta[i],Cmax[i],self.R[i],self.G[i],self.B[i])
			S[i] = getS(delta[i],Cmax[i],Cmin[i])
			L[i] = getL(Cmax[i],Cmin[i])
		self.HSL = np.vstack((H,S,L))

	def makeBW(self):
		# self.BW = 0.21*self.R + 0.72*self.G + 0.07*self.B   	#luminosity
		# self.BW = (self.R+self.G+self.B)/3.					#average
		self.BW = (np.amax(self.RGB,axis=0)+np.amin(self.RGB,axis=0))/2 #brightness

	def show(self):
		plt.figure()
		plt.imshow(self.RGB)
		plt.show()

	def showBreakdown(self,show=True):
		# i want to see the image, RGB together, breakdown of each color
		# breakdown into H,S,V,L categories
		f, axarr = plt.subplots(3,3,figsize=(9, 7))
		arrays = [self.R,self.G,self.B,
				  self.HSV[0],self.HSV[1],self.HSV[2],
				  self.HSL[0],self.HSL[1],self.HSL[2]]
		titles = ['R','G','B','H','S','V','H','S','L']
		cmaps = ['Reds','Greens','Blues','hsv','gray','gray',
				 'hsv','gray','gray']
		for i in range(9):
			axarr[i//3][i%3].imshow(self.makeImage(arrays[i]),cmap=cmaps[i])
			axarr[i//3][i%3].axis('off')
			axarr[i//3][i%3].set_title(titles[i])
		plt.tight_layout()
		plt.suptitle('Raw Image',fontsize=16,y=1.0)
		if show:
			plt.show()

	def showBreakdownByEdges(self,alg='Sobel',show=True):
		if alg == 'Sobel':
			detector = sobelOp
		elif alg == 'Canny':
			detector = canny
		else:
			print('This method is not supported')
			return

		f, axarr = plt.subplots(3,3,figsize=(9, 7))
		arrays = [self.R,self.G,self.B,
				  self.HSV[0],self.HSV[1],self.HSV[2],
				  self.HSL[0],self.HSL[1],self.HSL[2]]
		titles = ['R','G','B','H','S','V','H','S','L']
		for i in range(9):
			axarr[i//3][i%3].imshow(detector(self.makeImage(arrays[i]))[0],cmap='binary',vmin=0.,vmax=3.)
			axarr[i//3][i%3].axis('off')
			axarr[i//3][i%3].set_title(titles[i])
		plt.tight_layout()
		plt.suptitle('Processed with ' + alg + ' Edge Detection',fontsize=16,y=1.0)
		if show:
			plt.show()

	def makeImage(self,array):
		dim = self.dimensions
		if len(array) != 3:
			ret = np.copy(array)
			ret = np.reshape(ret,dim)
			return ret
		ret = np.empty((dim[0],dim[1],3))
		ret[:,:,0] = np.reshape(array[0],dim)
		ret[:,:,1] = np.reshape(array[1],dim)
		ret[:,:,2] = np.reshape(array[2],dim)
		return ret

class Curve():
	A = 0.0 #[0,1] bias of gradient measurements over local slope estimate
	def __init__(self,start,end,side):
		#direction is an angle that the curve will move towards
		self.start = np.array(start)
		self.end = np.array(end)
		self.l_tot = np.linalg.norm(self.end-self.start)
		self.v_old = (self.end-self.start)/self.l_tot #must be normalized
		self.c_tot = 0
		self.l_old = self.l_tot
		self.status = 'growing'
		self.side = side

	def update(self,new_pt,grad_new):
		##### assign directions to v_old & v_new
		new_pt = np.array(new_pt)
		vec_old_to_new = new_pt - self.end
		l_new = np.linalg.norm(vec_old_to_new)
		dir_old = np.arctan2(self.v_old[0],self.v_old[1])	
		dir_new = grad_new + self.side*np.pi/2.
		v_new = self.A*(vec_old_to_new/l_new) + \
				(1-self.A)*np.array([np.sin(dir_new),np.cos(dir_new)])
		th = np.arccos(np.dot(-self.v_old,v_new))
		denom = np.sqrt(self.l_old**2 + l_new**2 - 2*self.l_old*l_new*np.dot(-self.v_old,v_new))
		if abs(denom) > 0.01:
			c_new = 2*np.sin(th)/(denom)
			th_dif = dir_new%(2*np.pi) - dir_old%(2*np.pi)
			if th_dif < 0:
				c_new *= -1.0
		else:
			c_new = 0.0

		print('v_old: (%i,%i)' % tuple(self.v_old))
		print('l_new: %f' % l_new)
		print('old_direction: %f' % round(dir_old,3))
		print('vec_old_to_new: (%i,%i)' % tuple(vec_old_to_new[:]))
		print('new_direction: %f' % round(dir_new,3))
		print('v_new: (%f,%f)' % tuple(v_new[:]))
		print('theta: %f' % round(th,4))
		print('c_new: %f' % round(c_new,3))
		self.c_tot = (self.l_tot*self.c_tot + l_new*c_new)/(self.l_tot+l_new)
		self.l_tot += l_new
		self.l_old = l_new
		self.v_old = v_new
		self.end[:] = new_pt[:]

	def createPath(self):
		# this creates a continuous path
		if abs(self.c_tot) < 0.005:
			x = [self.start[0],self.end[0]]
			y = [self.start[1],self.end[1]]
			return np.array(x),np.array(y)
		r = 1./self.c_tot
		th = self.l_tot/r 
		n_segments = 25
		dth = th/n_segments

		#find the center of rotation: http://mathforum.org/library/drmath/view/53027.html
		mid_pt = (self.start+self.end)/2.
		vec = self.end-self.start
		q = np.linalg.norm(self.end-self.start)
		vec_norm = (self.end-self.start)/q
		th_rot = np.pi/2
		c,s = np.cos(th_rot),np.sin(th_rot)
		R_matrix = np.array(((c,-s), (s, c))).T
		rotated_vec = np.dot(R_matrix,vec_norm)
		CoR = mid_pt + np.sign(self.c_tot)*np.sqrt(r**2-(q/2.)**2)*rotated_vec
		n_points = 20
		c,s = np.cos(dth),np.sin(dth)
		dR_matrix = np.array(((c,-s), (s, c))).T
		x = [self.start[0]]
		y = [self.start[1]]
		pointing_vec = self.start-CoR	
		for i in xrange(n_points):
			pointing_vec = np.dot(dR_matrix,pointing_vec)
			x.append(CoR[0]+pointing_vec[0])
			y.append(CoR[1]+pointing_vec[1])
		return np.array(x),np.array(y)

	def __str__(self):
		h1 = '<'+'-'*25+'>\n'
		h2 = "INSTANCE of Curve OBJECT\n"
		l1 = "	starts at (%i,%i)\n" % (self.start[0],self.start[1])
		l2 = "	ends at (%i,%i)\n" % (self.end[0],self.end[1])
		l3 = "	length of %f \n" % self.l_tot
		l4 = "	curvature is %f \n" % round(self.c_tot,3)
		return h1+h2+l1+l2+l3+l4+h1

class Curve2():
	A = 0.0
	# double ended curve
	def __init__(self,seed,edges,gradients):
		self.rtail = np.array(seed)
		self.ltail = np.array(seed)
		self.rstatus = 'seeded'
		self.lstatus = 'seeded'
		self.status = 'growing'
		self.c_tot = 0.0
		self.l_tot = 0.0
		self.l_old = 0.0
		self.rconf = edges[seed]
		self.lconf = edges[seed]
		self.radius = np.inf
		self.edges = edges 
		self.grads = gradients
		self.AOIs = Curve2.getAOIs()

	def grow(self,pt,side_desc):
		grad = self.grads[pt]
		
		if side_desc =='right':
			side = 1
			status = self.rstatus
			tail = self.rtail
		elif side_desc == 'left':
			side = -1
			tail = self.ltail
			status = self.lstatus
		else:
			raise TypeError

		vec_lr = self.rtail-self.ltail
		q = np.linalg.norm(vec_lr)
		if status == 'growing':
			uvec_lr = vec_lr/q
			angle_prog = side*np.arcsin(q*self.c_tot/2.)
			c,s = np.cos(angle_prog),np.sin(angle_prog)
			R_matrix = np.array(((c,-s), (s, c))).T
			uvec_rl = side*uvec_lr 
			v_est = np.dot(R_matrix,uvec_rl)
		elif status == 'seeded':
			v_est = np.array([np.sin(grad+side*np.pi/2),np.cos(grad+side*np.pi/2)])

		pt = np.array(pt)
		new_dir = grad + side*np.pi/2
		vec_old_to_new = pt - tail
		l_new = np.linalg.norm(vec_old_to_new)
		old_dir = np.arctan2(v_est[0],v_est[1])
		v_new = self.A*(vec_old_to_new/l_new) + \
				(1-self.A)*np.array([np.sin(new_dir),np.cos(new_dir)])
		th = np.arccos(np.dot(-v_est,v_new))
		denom = np.sqrt(self.l_old**2 + l_new**2 - 2*self.l_old*l_new*np.dot(-v_est,v_new))
		if abs(denom) > 0.01:
			c_new = 2*np.sin(th)/(denom)
			th_dif = new_dir%(2*np.pi) - old_dir%(2*np.pi)
			if th_dif < 0:
				c_new *= -1.0
		else:
			c_new = 0.0
		if th > np.pi/2:
			th = np.pi-th
		if abs(th) > np.pi/6:
			new_status = 'dormant'
			print('went dormant')
		else:
			new_status = 'growing'
			self.c_tot = (self.l_tot*self.c_tot + side*l_new*c_new)/(self.l_tot+l_new)
			if 1/self.c_tot < q/2:
				self.c_tot = 2./q
			self.radius = 1/self.c_tot if abs(self.c_tot) > 0.0001 else np.inf
			self.l_tot += l_new
			self.l_old = l_new


		if side_desc == 'right':
			self.rtail = pt[:]
			self.rstatus = new_status
		else:
			self.ltail = pt[:]
			self.lstatus = new_status
		
	def rgrow(self,rseed):
		return self.grow(rseed,'right')

	def lgrow(self,lseed):
		return self.grow(lseed,'left')

	def path(self):
		# go from left to right
		if abs(self.c_tot) < 0.005:
			x = [self.ltail[0],self.rtail[0]]
			y = [self.ltail[1],self.rtail[1]]
			return np.array(x),np.array(y)

		#find the center of rotation: http://mathforum.org/library/drmath/view/53027.html
		mid_pt = (self.ltail+self.rtail)/2.
		vec = self.rtail-self.ltail
		q = np.linalg.norm(self.rtail-self.ltail)
		r = self.radius
		th = 2*np.arcsin(min(q*self.c_tot/2.,1.0)) 
		n_segments = 20
		dth = th/n_segments
		vec_norm = (self.rtail-self.ltail)/q
		th_rot = np.pi/2
		c,s = np.cos(th_rot),np.sin(th_rot)
		R_matrix = np.array(((c,-s), (s, c))).T
		rotated_vec = np.dot(R_matrix,vec_norm)
		CoR = mid_pt + np.sign(self.c_tot)*np.sqrt(r**2-(q/2.)**2)*rotated_vec
		n_points = 20
		c,s = np.cos(dth),np.sin(dth)
		dR_matrix = np.array(((c,-s), (s, c))).T
		x = [self.ltail[0]]
		y = [self.ltail[1]]
		pointing_vec = self.ltail-CoR	
		for i in xrange(n_points):
			pointing_vec = np.dot(dR_matrix,pointing_vec)
			x.append(CoR[0]+pointing_vec[0])
			y.append(CoR[1]+pointing_vec[1])
		return np.array(x),np.array(y)

	def expand(self):
		seeds = []
		if self.rstatus != 'dormant':
			r_dir = self.grads[tuple(self.rtail)] + np.pi/2
			r_AOI_id = Curve2.selectAOI(r_dir)
			r_AOI = self.AOIs[r_AOI_id]
			rseed,_ = self.sampleAOI(self.rtail,r_AOI)
			self.rgrow(rseed)
			seeds.append(rseed)
		if self.lstatus != 'dormant':
			l_dir = self.grads[tuple(self.ltail)] - np.pi/2
			l_AOI_id = Curve2.selectAOI(l_dir)
			l_AOI = self.AOIs[l_AOI_id]
			lseed,_ = self.sampleAOI(self.ltail,l_AOI)
			self.lgrow(lseed)
			seeds.append(lseed)
		return seeds

	def sampleAOI(self,c,AOI):
		edges = self.edges
		ci,cj = c
		AOI_radius = AOI.shape[0]//2
		AOI_len = AOI.shape[0]
		w,h = edges.shape
		w_gap = w-ci-1
		h_gap = h-cj-1
		#so we dont have indexing error
		AOI_trimmed = AOI[max(AOI_radius-ci,0):min(w_gap+AOI_radius+1,AOI_len),
						  max(AOI_radius-cj,0):min(h_gap+AOI_radius+1,AOI_len)]
		trimmed_h = AOI_trimmed.shape[1]
		subsection = edges[ci-min(ci,AOI_radius):ci+1+min(w_gap,AOI_radius),
						   cj-min(cj,AOI_radius):cj+1+min(h_gap,AOI_radius)]
		sample = np.multiply(subsection,AOI_trimmed)
		best_id = np.argmax(sample)
		best_i = best_id//sample.shape[1] + ci-min(ci,AOI_radius)
		best_j = best_id % sample.shape[1] + cj-min(cj,AOI_radius)
		rating = 0 # will implement this later
		return (best_i,best_j), rating
	
	def __str__(self):
		h1 = '<'+'-'*25+'>\n'
		h2 = "INSTANCE of Curve OBJECT\n"
		l1 = "	left end at (%i,%i)\n" % (self.ltail[0],self.ltail[1])
		l2 = "	right end at (%i,%i)\n" % (self.rtail[0],self.rtail[1])
		l3 = "	length of %f \n" % self.l_tot
		l4 = "	curvature is %f \n" % round(self.c_tot,4)
		l5 = "	est. radius is %f \n" % self.radius
		return h1+h2+l1+l2+l3+l4+l5+h1

	@staticmethod
	def getAOIs():
		# 5x5 matrices centered on location to take weighted average of the best edge
		# direction is an angle that represents the likely location of an edge
		# indexed 0-7: right to bottom right in CCW direction
		side = np.zeros((5,5))
		# side[2,3] = 1
		side[:,4] = 1
		corner = np.zeros((5,5))
		corner[0,2:5] = 1
		corner[1,3:5] = 1
		corner[2,4] = 1
		AOIs = np.zeros((8,5,5))
		for i in range(8):
			if i % 2 == 0: #side
				AOIs[i] = side[:,:]
				side = np.rot90(side)
			else:
				AOIs[i] = corner[:,:]
				corner = np.rot90(corner)
		return AOIs	

	@staticmethod
	def selectAOI(direction):
		#returns the index corresponding to the right AOI
		# direction is an angle
		direction = np.fmod(-direction,2*np.pi)
		if direction < 0:
			direction += 2*np.pi
		#now direction is greater than 0
		index = (direction +np.pi/8)//(np.pi/4)
		return int(index)%8


##############
##############
#
#	FILTERS
# 
#############
#############

def getEdges(img,o):
	k = getEdgeKernel(7,o)
	edges = ndimage.convolve(img,k,mode='constant',cval=0.0)
	return edges

def sobelOp(img):
	Kx = np.array([[1.0,0,-1.0], [2.0,0.0,-2.0], [1.0,0.0,-1.0]])
	Ky = np.array([[1.0,2.0,1.0], [0.0,0.0,0.0], [-1.0,-2.0,-1.0]])
	Gx = ndimage.convolve(img,Kx,mode='nearest')
	Gy = ndimage.convolve(img,Ky,mode='nearest')
	edges = np.sqrt(np.multiply(Gx,Gx)+np.multiply(Gy,Gy))
	gradients = np.arctan2(Gy,Gx)
	return edges, gradients

def canny(img):
	# header
	return feature.canny(img,sigma=3)

def smoothImage(img):
	# kernel = np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])
	# kernel = kernel / np.sum(kernel)
	# smoothed = ndimage.convolve(img,kernel,mode='constant',cval=0.0)
	return img

def smoothCurves(img, kernels=None,ksize=5,sigma=3):
	if kernels is None:
		G, Gdot, Gddot = getCurveSmootherKernels(ksize,sigma)
	return ndimage.convolve(img,smoother,mode='constant',cval=0.0)

def sharpen(img):
	k = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]])
	return ndimage.convolve(img,k,mode='constant',cval=0.0)

def blur(img):
	k = np.ones((3,3))/9.
	return ndimage.convolve(img,k,mode='nearest')

def watershedSeg(img,nsteps,scope=1):
	#img must be image array
	w,h = img.shape
	cumul = np.zeros(img.shape)
	old_step = np.ones(img.shape)
	new_step = np.zeros(img.shape)

	def get_neighbors(scope=1):
		#scope is how far out you can put your waters
		d_neighbors = []
		for i in range(-scope,scope+1):
			for j in range(-scope,scope+1):
				if i!=j:
					d_neighbors.append((i,j))
		return d_neighbors

	d_neighbors = get_neighbors(scope)
	grad_neighbors = np.zeros(len(d_neighbors))
	for step_id in range(nsteps):
		for index in xrange(img.size):
			i,j = i,j = index//h, index % h
			sum_grad = 0
			if old_step[i,j] > 0.0001:
				for n_id in xrange(len(d_neighbors)):
					di,dj = d_neighbors[n_id]
					try: #to get gradient
						grad = img[i,j]-img[(i+di),(j+dj)]
						grad_neighbors[n_id] = grad
						if grad > 0: 
							sum_grad += grad
					except IndexError:
						grad_neighbors[n_id] = 0.0
				if sum_grad > 0:
					for n_id in xrange(len(d_neighbors)):
						di,dj = d_neighbors[n_id]
						if grad_neighbors[n_id] > 0:
							flow = old_step[i,j] * grad_neighbors[n_id]/sum_grad
							new_step[(i+di),(j+dj)] += flow
							cumul[(i+di),(j+dj)] += flow
		old_step = new_step
		new_step = np.zeros(img.shape)
	return cumul

def findCurves(edges,gradients):

	curveMap = None

	curveList = None

def edgeSniffer(edges,grouping=40,style='relative'):
	#take an image of edge likelihoods
	# finds the best edge candidate in a section of grouping^2 pixels
	# returns a list of indeces 
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
	indices = indices[np.where(maxes > 0.2)]

	return indices

def localConnector(curve1,curve2):
		# when two curves become very close to each other
		pass

def getEdgeKernel(size, orientation):
	# size must be odd
	if size % 2 == 0:
		size += 1

	c = size//2 
	k = np.zeros((size,size))
	vec = np.array([np.cos(orientation*np.pi/8),
					 np.sin(orientation*np.pi/8)])
	num_pos = 0
	for i in range(size):
		dy = i-c
		for j in range(size):
			dx = j-c
			# find projected distances along lines
			a1 = np.dot(np.array([dx,dy]),vec)
			a2_vec = np.array([dx,dy]) - a1*vec
			a1 = abs(a1)
			a2 = np.sqrt(a2_vec[0]**2+a2_vec[1]**2)
			val = 2**(-a1)
			if a2 < 0.5:
				k[i][j] = 4*val
				num_pos +=1
			if 0.5 <= a2 < 1.5:
				k[i][j] = - 2*val

	total_sum = sum(sum(k))
	if abs(total_sum) > 0.01:
		for i in range(size):
			for j in range(size):
				if k[i][j] > 0:
					k[i][j] -= total_sum/num_pos
	return k

def singleEdgeFinder(loc,edges,gradients,stepwise=True,verbose=True):
	# edges is an image array, lowerbnd of 0, 
	# higher number is more likely to be an edge
	w,h = edges.shape
	max_i,max_j = loc
	rgba = plt.cm.gray(edges/np.amax(edges))
	rgba[max_i,max_j] = 1,0,0,1
	
	seed = max_i,max_j
	lseed = seed[:]
	rseed = seed[:]
	curve = Curve2(seed,edges,gradients)
	print(curve)

	plt.figure(figsize=(9,7))
	width = 40
	plt.imshow(rgba[max_i-width:max_i+width,max_j-width:max_j+width],
				extent = [max_j-width,max_j+width,max_i+width,max_i-width])
	plt.autoscale(False)
	path, = plt.plot([],[],'b-')
	plt.tight_layout()
	n_steps = 16
	for i in xrange(n_steps):
		seeds = curve.expand()
		for s in seeds:
			rgba[s] = 0,0,1,1
		if stepwise:
			px,py = curve.path()
			path.set_data(py,px)
			plt.title('after %i steps' % (i+1))
			plt.draw()
			plt.pause(0.15)
		if verbose:
			print(curve)
	px,py = curve.path()
	path.set_data(py,px)
	plt.show()

def multiEdgeFinder(edges,gradients,grouping=50,n_steps=5):
	edges *= np.random.normal(1.0,0.05,size=edges.shape)
	gradients *= np.random.normal(1.0,0.05,size=edges.shape)
	plt.figure(figsize=(9,7))
	rgba = plt.cm.gray(edges/np.amax(edges))
	seeds = edgeSniffer(edges,grouping)
	curves = []
	curve_plots = []

	for s in seeds:
		curves.append(Curve2(s,edges,gradients))
		c_plot, = plt.plot([],[],'b-')
		curve_plots.append(c_plot)
		rgba[tuple(s)] = 1,0,0,1

	plt.imshow(rgba)
	plt.autoscale(False)
	plt.title('Global Edge Finder: Iteration 0')
	plt.tight_layout()

	for n in range(n_steps):
		for i,c in enumerate(curves):
			if c.status == 'growing':
				c.expand()
				px,py = c.path()
				curve_plots[i].set_data(py,px)
		plt.title('Global Edge Finder: Iteration %i' % (n+1))
		plt.draw()
		plt.pause(0.2)
	plt.show()

###################
#################
##################

if __name__ == "__main__":
	img = Image('simple_edges')
	# img.showBreakdown(show=False)
	# img.showBreakdownByEdges()
	# plt.figure(figsize=(9,7))
	S_img = img.makeImage(img.HSL[1])
	edges,gradients = sobelOp(S_img)
	# plt.imshow(blur(edges),cmap='gray',vmin=0.,vmax = 1.)
	# plt.tight_layout()

	# multiEdgeFinder(edges,gradients,n_steps=5)
	loc = 48,67
	singleEdgeFinder(loc,edges,gradients)


# # resources
# colormap, color maps: https://matplotlib.org/examples/color/colormaps_reference.html

# curve smoothing: https://www.computer.org/csdl/proceedings/cvpr/1988/0862/00/00196255.pdf

# watershed: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1634338


##########################

# try applying v_est during AOI selection, to allow the curve to 'jump over' poor data

# implement an energy for each tail.  use bayes: we have the value Edges(x,y) of 'energy'
#		added to a tail when it absorbs a new point, and the proportion of 'energy' 
#		available to the tail is proportional to the difference between the direction
#		of the tail (using v_est) and the direction of the new point (using v_new)
# create a way to visualize the energy throughout the curve
# allow the curves to overextend but then shrink back when it realizes it has gone too far
# 		without discovering any new energy
# pair up 
# the left and right tails should share the 'energy' in some manner while maintaining knowledge
#		of their contributions, if they are being fed then they should be less likely to take a
#		risk in extending themselves

"""
The curve representation:
	-constant memory
	-stepwise update rule
	-globally specific for comparion/joining
	-nondiscrete predictions


"""



""" 
lets break it down into problems, then find an order, and then a plan

Task 1. We need to create a map(s) of edge likelihood (and maybe a gradient field)

Task 2. We need a strategy to pick a potential edge to model

Task 3. We need a way to transform a geometric scattering of edge likelihoods into a 'curve'

Task 4. We need a way to represent the curves in the program

Task 5. We need a way to judge the truth/probability that an curve exists

Task 6. We need a way to join curves that touch/ or decide that there is a curve termination

Task 7. We need a way to join curves globally (for non-flat curves, cluster by center/radius,
			for flat curves, cluster by slope/height ), and identify occlusion

Task 8. We need a way to determine if a corner exists

Task 9. We need to be able to join curves into a shape

Task 10. We need a way to determine what light channels are important (conditional on 
		macro-data of the image?) and how to combine knowledge from multiple channels

Task 11. We need a coordinate system that is more reliable (takes into account image size, 
		focal length, etc)
"""