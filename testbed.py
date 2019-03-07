import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('classic')

def gaussian(x, mu, var):
	return np.exp(-np.power(x-mu, 2.)/(2.*var))

DATA = np.loadtxt('data.txt', delimiter = ',')
DATA = DATA[DATA[:, 0].argsort()]
ids = DATA[:,0]
x_coords = DATA[:,1]
y_coords = DATA[:,2]
d_angles_real = DATA[:,3]*np.sign(DATA[:,0])



vecs = DATA[1:,1:3]-DATA[:-1,1:3]
d_angles = np.zeros(vecs.size-1)

angles = np.arctan2(vecs[:,0],vecs[:,1])
d_angles = angles[1:] - angles[:-1]
d_angles[d_angles >  np.pi] -= 2*np.pi
d_angles[d_angles < -np.pi] += 2*np.pi
d_angles /= np.pi/4. # discretize

thetas = np.cumsum(d_angles)

fig = plt.figure(figsize=(16,10))
plt.tight_layout()
ax0 = fig.add_subplot(411)
ax0.plot(thetas, '-o', markersize=3)
ax0.set_ylim((np.amin(thetas)-1,np.amax(thetas)+1))

THRESHOLD_ERR = 0.6
ax1 = fig.add_subplot(412)
# ax1.plot(thetas, '-o', markersize=3)
# ax1.set_ylim((np.amin(thetas)-1,np.amax(thetas)+1))
err_plot, = ax1.plot([],[], '-o', markersize=3)
ax1.plot([0,thetas.size],[0,0],'k--')
ax1.plot([0,thetas.size],2*[THRESHOLD_ERR],'k:')
ax1.plot([0,thetas.size],2*[-THRESHOLD_ERR],'k:')
ax1.set_ylim((-2.5,2.5))


ax2 = fig.add_subplot(413)
lag_plot, = ax2.plot([],[])
front_plot, = ax2.plot([],[])
pred_plot, = ax2.plot([],[], '-x', markersize=4)
# ax2.legend(['average', 'observed', 'predicted'])
ax2.plot([0,thetas.size],[0,0],'k--')
ax2.set_ylim((-2.5,2.5))
ax2.set_yticks(np.arange(-2,3))

ax3 = fig.add_subplot(414)
ax3.plot([0, thetas.size], 2*[0], 'k:')
ax3.plot(d_angles, '-o', markersize=3)
ax3.set_ylim((-2.5,2.5))


LAG = []
FRONT = []
ERR = []
I = []
CORNERS = []


err = 0
count = 0
total_i = 0
x_arr = np.zeros(3)
total_i = 0
running = True
while running:
	err = 0
	count = 0
	x_arr = np.zeros(3)
	while True:
		I.append(total_i)
		if count == 0:
			x_front = 0
			mu_front = 0
		else:
			x_front += d_angles[total_i]
			mu_front += (x_front-mu_front)/2.

		if count < 3:
			x = 0
			mu = 0
		else:
			x += d_angles[total_i-3]
			mu += (x - mu)/(count-2.0)

		err += (mu - mu_front)
		err *= 0.8
		FRONT.append(mu_front)
		LAG.append(mu)
		ERR.append(err)
		front_plot.set_data(I,FRONT)
		lag_plot.set_data(I, LAG)
		err_plot.set_data(I,ERR)
		
		if total_i == thetas.size-1:
			running = False
			break
		if abs(err) > THRESHOLD_ERR:	
			ax0.plot(total_i, thetas[total_i], 'r|', markersize=200)
			ax1.plot(total_i, 0, 'r|', markersize=200)
			ax2.plot(total_i, 0, 'r|', markersize=200)
			ax3.plot(total_i, 0, 'r|', markersize=200)
			CORNERS.append(total_i)
			break
		count += 1
		total_i += 1


plt.figure()
plt.plot(y_coords,-x_coords)
plt.plot(y_coords[ids==0], -x_coords[ids==0], 'g^', markersize=6)
plt.plot(y_coords[CORNERS], -x_coords[CORNERS], 'rs', markersize=3)
plt.axis('equal')

plt.show()
