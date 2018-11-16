import numpy as np 
import numpy.random as npr 

class Neuron:
	def __init__(self, nCnxns):
		self.nCnxns = nCnxns
		self.resetMapping()
		self.permanence = np.full(nCnxns, 0.5)
		self.permThresh = 0.45
		self.alpha = 0.01

	def resetMapping(self):
		self.mapping = 2*npr.randint(2,size=self.nCnxns)-1

	def activeMapping(self):
		ret = np.copy(self.mapping)
		ret[self.permanence < self.permThresh] = 0
		return ret

	def processInfo(self, info):
		return int(np.dot(info, self.activeMapping()) >= 0)

	def updatePermanence(self, info, didFire):
		changes = self.alpha*(2*info - 1)
		if didFire:
			self.permanence += changes
		else:
			self.permanence -= changes
		self.permanence = np.clip(self.permanence/np.sum(self.permanence), 0.0, 1.0)

class DataSet:
	def __init__(self, size):
		self.size = size
		self.log = []
		self.dist = np.zeros(size,dtype=int)

	def generateInfo(self):
		if npr.random() > 0.5:
			ret = npr.randint(2, size=self.size)
		else:
			ret = np.zeros(self.size,dtype=int)
		if npr.random() > 0.1:
			ret[-1] = 1
		self.updateLog(ret)
		self.updateDist(ret)
		return ret

	def updateLog(self,info):
		self.log.append(info)

	def updateDist(self,info):
		self.dist += info

	def showLog(self,complete=True):
		if complete:
			for row in self.log:
				print([a for a in row])
			print('------------')
		print([a for a in self.dist])



size = 8
dataset = DataSet(size)
neuron = Neuron(size)

ntrials = 100
for _ in xrange(ntrials):
	info = dataset.generateInfo()
	output = neuron.processInfo(info)
	string1 = ', '.join([str(i) for i in info])
	string2 = ', '.join([str(round(i,2)) for i in neuron.permanence])
	print('(%s) -> %i : (%s)' % (string1, output, string2) )
	neuron.updatePermanence(info, bool(output))

dataset.showLog(False)
print(neuron.mapping)
