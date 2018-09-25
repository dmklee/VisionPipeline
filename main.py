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

class Eye():
	def __init__(self,image):
		self.image = image
		self.edges,self.gradients = sobelOp(self.image)
		self.edges = self.edges/np.amax(self.edges)
		self.gradients = -self.gradients
		self.origin = np.array(self.image.shape,dtype=int)/2
		self.rxmax = self.origin[1]
		self.rymax = self.origin[0]

	def edgeR(self,rx,ry):
		px,py = realToPix(rx,ry)
		return self.EdgeP(px,py)

	def edgeP(self,px,py):
		return self.edges[px,py]

	def rgradient(self,rx,ry):
		px,py = realToPix(rx,ry)
		return self.gradientP(px,py)

	def pgradient(self,px,py):
		return self.gradients[px,py]

	def Pgradient(self,PXY):
		return self.gradients[PXY[:,0],PXY[:,1]]

	def Rgradient(self,RXY):
		PXY = self.realToPixArray(RXY)
		return self.Pgradient(PXY)

	def pixToReal(self,px,py):
		rx = py-self.origin[1]
		ry = self.origin[0] - px
		return rx,ry

	def pixToRealArray(self,PXY):
		RXY = np.flip(np.multiply(PXY.astype(np.float32)-self.origin,np.array([-1,1])),1)
		return RXY

	def realToPix(self,rx,ry):
		px = int(self.origin[0]-ry)
		py = int(rx+self.origin[1])
		return px,py

	def realToPixArray(self,RXY,rounded=True):
		PXY = (np.multiply(np.flip(RXY,1),np.array([-1,1]))+self.origin)
		if rounded:
			PXY = PXY.astype(int)
		return PXY

	def see(self,complete=True):
		plt.figure(figsize=(12,8))
		# plt.xticks(np.arange(2*self.rxmax+1,step=self.rxmax/2-0.2),
		# 		[str(-self.rxmax),str(-self.rxmax/2),'0',str(self.rxmax/2),str(self.rxmax)])
		# plt.yticks(np.arange(2*self.rymax+1,step=self.rymax/2-0.2),
		# 		[str(-self.rymax),str(-self.rymax/2),'0',str(self.rymax/2),str(self.rymax)])
		rgba = plt.cm.gray(self.edges)
		# rgba[tuple(self.origin)] = 1,0,0,1
		plt.imshow(rgba)
		plt.autoscale(False)
		plt.tight_layout()
		if complete:
			plt.show()			

	def findCurves(self,p_loc=(107,133),anim=True):
		self.see(False)
		plt.plot(p_loc[1],p_loc[0],'r*')
		path_plt, = plt.plot([],[],'b-')
		for step_num in range(5):
			if step_num == 0:
				curve = Curve(p_loc,self)
			else:
				curve.expand()
			print(curve)
			path = self.realToPixArray(curve.rpath(),rounded=False)
			path_plt.set_data(path[:,1],path[:,0])
			plt.title("Curve Growth after %i expansions" % step_num)
			if anim:
				plt.draw()
				plt.pause(0.5)
			# new_pts = self.pixToRealArray(np.array((curve.pRtail,curve.pLtail)))
			# plt.plot(new_pts[:,0],new_pts[:,1],'r*')
			# thetas = self.Pgradient(np.array((curve.pRtail,curve.pLtail)))
			# for i in range(2):
			# 	plt.plot([new_pts[i,0],new_pts[i,0]+2*np.cos(thetas[i])],
			# 			[new_pts[i,1],new_pts[i,1]+2*np.sin(thetas[i])],'k-')
		plt.show()

class Curve():

	def __init__(self,pseed,eye):
		self.curv = 0.0
		self.eye = eye
		self.tilt = self.eye.pgradient(pseed[0],pseed[1])
		self.pseed = tuple(pseed)
		self.pLtail = tuple(pseed)
		self.pRtail = tuple(pseed)
		self.length = 0
		self.AOIs = Curve.getAOIs()

	def expand(self):
		if np.random.uniform() > 0.5:
			self.grow('right')
			self.grow('left')
		else:
			self.grow('left')
			self.grow('right')

	def getSideInfo(self,side_desc):
		if side_desc == 'right':
			side = -1
			ptail = self.pRtail
		else:
			side = 1
			ptail = self.pLtail
		return side, ptail

	def setSideInfo(self,side_desc,ptail):
		if side_desc == 'right':
			self.pRtail = ptail
		else:
			self.pLtail = ptail

	def updateTilt(self,new_th):
		diff = angleDiff(new_th,self.tilt)
		self.tilt += diff/(self.length)

	def getCnew(self,pt,side_desc):
		side,ptail = self.getSideInfo(side_desc)
		vec_StoNew = np.array(pt)-np.array(self.pseed)
		th_StoNew = np.arctan2(vec_StoNew[1],vec_StoNew[0])
		q = np.linalg.norm(vec_StoNew)

		#find the curvature given the location of the new point
		alpha = self.tilt-th_StoNew
		if abs(abs(alpha) - np.pi/2) < 0.0001:
			c_loc = 0.0
		else:	
			r_loc = q/(2*np.cos(abs(alpha)))
			c_loc = side*np.sign(alpha)/r_loc
		c_loc = np.clip(c_loc,-0.1,0.1)
		# find the curvature based on the gradient of the new point
		new_grad = self.eye.pgradient(pt[0],pt[1])
		beta = angleDiff(new_grad,self.tilt)
		if beta%np.pi == 0:
			c_grad = 0.0
		else:
			c_grad = -side*np.sign(beta)*2*np.sin(abs(beta)/2.)/q
		c_grad = np.clip(c_grad,-0.1,0.1)

		z = self.length/10
		z = np.clip(z,0,1)
		c_new = (z*c_loc+(1-z)*c_grad)/2
		return c_new

	def getAngleProg(self,vec_StoTail,q):
		if abs(self.curv) < 0.0001 or q == 0:
			return 0
		th_StoTail = abs(np.arctan2(vec_StoTail[1],vec_StoTail[0]))
		alpha = angleDiff(self.tilt,th_StoTail)
		return np.sign(alpha)*(np.pi-2*abs(alpha))

	def getModeledTail(self,side,angleProg):
		# only works if angleProg != 0
		q_model = 2.0/abs(self.curv)*np.sin(angleProg/2.)
		alpha = (np.pi-angleProg)/2.
		model_tail = q*np.array((np.cos(self.tilt+side*alpha),np.sin(self.tilt+side*alpha)))
		return model_tail

	def updateCurv(self,pt,side_desc):
		c_new = self.getCnew(pt,side_desc)
		self.curv += (c_new-self.curv)/(self.length)

	def grow(self,side_desc):
		side,ptail = self.getSideInfo(side_desc)
		tail_dir = self.eye.pgradient(ptail[0],ptail[1])+side*np.pi/2.
		AOI = self.AOIs[Curve.selectAOI(tail_dir)]
		new_pt = self.sampleAOI(ptail,AOI)
		self.length += np.linalg.norm(np.array(new_pt)-np.array(ptail))
		new_th = self.eye.pgradient(new_pt[0],new_pt[1])
		self.updateTilt(new_th)
		self.updateCurv(new_pt,side_desc)
		self.setSideInfo(side_desc,new_pt)

	def rpath(self):
		rseed = np.array(self.eye.pixToReal(self.pseed[0],self.pseed[1])).astype(float)
		if abs(self.curv) < 0.00001:
			rpath = np.stack((rseed,rseed))
			rpath[:,0] += np.array([-5,5])*np.cos(self.tilt+np.pi/2)
			rpath[:,1] += np.array([-5,5])*np.sin(self.tilt+np.pi/2)
			return rpath
		radius = 1/self.curv
		rcenter = rseed + radius*np.array((np.cos(self.tilt),np.sin(self.tilt)))
		vec_CtoS = rseed - rcenter
		holding_vec = rseed-rcenter
		dth = np.pi/50
		c,s = np.cos(dth),np.sin(dth)
		dR_matrix = np.array(((c,-s), (s, c)))
		output = np.empty((40,2))
		for i in xrange(20,40):
			holding_vec = np.dot(dR_matrix,holding_vec)
			output[i,:] = holding_vec[:]
		holding_vec = rseed-rcenter
		c,s = np.cos(-dth),np.sin(-dth)
		dR_matrix = np.array(((c,-s), (s, c)))
		for i in np.arange(20)[::-1]:
			holding_vec = np.dot(dR_matrix,holding_vec)
			output[i,:] = holding_vec[:]
		output += rcenter + 0.5
		return output

	def sampleAOI(self,c,AOI):
		edges = self.eye.edges
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
		return best_i,best_j

	def __str__(self):
		h1 = '<'+'-'*25+'>\n'
		h2 = "INSTANCE of Curve OBJECT\n"
		l1 = "	left end at (%i,%i)\n" % (self.pLtail[0],self.pLtail[1])
		l2 = "	right end at (%i,%i)\n" % (self.pRtail[0],self.pRtail[1])
		l3 = "	seed at (%i,%i)\n" % (self.pseed[0],self.pseed[1])
		l4 = "	curvature is %f \n" % round(self.curv,4)
		l5 = "	tilt is %f \n" % self.tilt
		return h1+h2+l1+l2+l3+l4+l5+h1
		pass

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
		direction = direction%(2*np.pi)
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

def guassianFilter(img,sigma=1.0):
	return ndimage.filters.gaussian_filter(img,sigma)

def scharrOp(img):
	Kx = np.array([[3.0,0,-3.0], [10.0,0.0,-10.0], [3.0,0.0,-3.0]])
	Ky = np.array([[3.0,10.0,3.0], [0.0,0.0,0.0], [-3.0,-10.0,-3.0]])
	Gx = ndimage.convolve(img,Kx,mode='nearest')
	Gy = ndimage.convolve(img,Ky,mode='nearest')
	edges = np.sqrt(np.multiply(Gx,Gx)+np.multiply(Gy,Gy))
	gradients = np.arctan2(Gy,Gx)
	return edges, gradients

def prewittOp(img):
	Kx = np.array([[1.0,0,-1.0], [1.0,0.0,-1.0], [1.0,0.0,-1.0]])
	Ky = np.array([[1.0,1.0,1.0], [0.0,0.0,0.0], [-1.0,-1.0,-1.0]])
	Gx = ndimage.convolve(img,Kx,mode='nearest')
	Gy = ndimage.convolve(img,Ky,mode='nearest')
	edges = np.sqrt(np.multiply(Gx,Gx)+np.multiply(Gy,Gy))
	gradients = np.arctan2(Gy,Gx)
	return edges, gradients

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

def edgeSniffer(edges,grouping=40,style='relative'):
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
	n_steps = 25
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
	img = Image('simple_edges2')
	S_img = img.makeImage(img.HSL[1])
	eye = Eye(S_img)
	eye.findCurves()


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