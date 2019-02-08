import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use('classic')

data = np.loadtxt('data.txt', delimiter = ',')#[::-1]
index = int(data[0])
data = data[1:]
# data = np.abs(data)
# data[data > 2.] -= 2*np.pi
# if np.amax(data) == 7.5:
# 	i = np.argmax(data)
# 	data[i:] -= 8
# 	# data[data == -8] = 0

fig = plt.figure(figsize=(8,10))
ax = fig.add_subplot(311)
N = 1
# plt.ylim((-8,8))
plt.plot(np.convolve(data, np.ones(N)/N, mode='valid'), '.-')
plt.plot(index, np.convolve(data, np.ones(N)/N, mode='valid')[index], 'r.', markersize=10)
plt.title('Angle')
# plt.show()

fig.add_subplot(312)
N = 1
data = data[1:]-data[:-1]
plt.plot(np.convolve(data, np.ones(N)/N, mode='valid'), '.-')
plt.plot(index-1, np.convolve(data, np.ones(N)/N, mode='valid')[index-1], 'r.', markersize=10)
plt.title('dAngle (raw)')


fig.add_subplot(313)
N = 1
data = data[1:]-data[:-1]
plt.plot(np.convolve(data, np.ones(N)/N, mode='valid'), '.-')
plt.plot(index-2, np.convolve(data, np.ones(N)/N, mode='valid')[index-2], 'r.', markersize=10)
plt.title('ddAngle (raw)')
plt.show()