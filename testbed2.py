import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('classic')

DATA = np.loadtxt('data_07.txt', delimiter = ',')

DATA = DATA[DATA[:, 0].argsort()]
THETA = np.cumsum(DATA[:,2])
plt.figure()
plt.plot(DATA[:,0], THETA,'k-')
plt.show()