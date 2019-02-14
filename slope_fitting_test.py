import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt 
from matplotlib import animation
plt.style.use('classic')

X = np.arange(0, 30, 0.3)
Y = 0.5*X + npr.random(X.size)

plt.figure()
plt.plot(X,Y)
plt.show()