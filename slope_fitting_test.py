import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt 
from matplotlib import animation
plt.style.use('classic')

X = np.arange(0, 30, 0.3)
Y = 0.5*X + npr.random(X.size)

def f(x,y):
	return 18.83599*x + 1.*y -2268.64

x = np.array([112,112,112,112,112,111,111,111,111,111,111])
y = np.array([159,160,161,162,163,164,165,166,167,168,169])

plt.figure()
# plt.plot(-(np.arange(159,170) -2268.64)/18.83599, np.arange(159,170))
plt.plot(x,y,'b.')
plt.ylim((150,175))
plt.xlim((107,115))
plt.show()