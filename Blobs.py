import numpy as np 
import matplotlib.pyplot as plt 

class Blob:
	def __init__(self, loc, size= 1):
		self.loc = np.array(loc)
		self.size = size

class Iline:
	def __init__(self, blob_1, blob_2):
		self.blob_1 = blob_1
		self.blob_2 = blob_2
		self.dir = blob_2.loc - blob_1.loc
		self.len = np.linalg.norm(self.dir)
		self.dir /= self.len
		self.prob = 1.0

def showBlobs(blobs,show=True):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for blob in blobs:
		plt.plot(blob.loc[0], blob.loc[1],'k.', markersize=10*blob.size)

	plt.xlim((0,1))
	plt.ylim((0,1))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	if show:
		plt.show()

def findBlobs(image=None):
	b1 = Blob((0.23, 0.46))
	b2 = Blob((0.33, 0.56))
	return [b1,b2]

def 

def main():
	b1 = Blob((0.23, 0.46))
	showBlobs([b1])





if __name__ == "__main__":
	main()