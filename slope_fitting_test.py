import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt 
from matplotlib import animation
plt.style.use('classic')

DATA = np.loadtxt('data.txt', delimiter=',')

data_sorted = DATA[DATA[:, 0].argsort()]

plt.figure()
plt.plot(data_sorted[:,0], data_sorted[:,1],'.')
plt.show()