import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('classic')

def gaussian(x, mu, var):
	return np.exp(-np.power(x-mu, 2.)/(2.*var))

DATA = np.loadtxt('data.txt', delimiter = ',')
DATA = DATA[DATA[:, 0].argsort()][::-1][5:250]
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

fig = plt.figure(figsize=(16,6))
ax1 = fig.add_subplot(411)
plt.plot(d_angles, '-o', markersize=3)
plt.tight_layout()
plt.ylim((np.amin(d_angles)-1, np.amax(d_angles)+1))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('dAngle')

ax2 = fig.add_subplot(412, sharex=ax1)
theta = np.cumsum(d_angles).astype(int)
plt.plot(theta, '-o', markersize=3)
plt.tight_layout()
plt.ylim((np.amin(theta)-1, np.amax(theta)+1))
plt.yticks(np.arange(np.amin(theta)-1,np.amax(theta)+2))
plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Angle')

ax3 = fig.add_subplot(413, sharex=ax1)
ax3_data = []
ax3_plot, = plt.plot([],[])
plt.tight_layout()
plt.ylabel("Uncertainty")
plt.ylim((-0.5, 0.5))

# ax4 = fig.add_subplot(414)
# PDF_prior = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
# PDF = np.copy(PDF_prior)
# PDF_bar = plt.bar([a-2 for a in range(5)], PDF, width=1, align='center')
# plt.tight_layout()
# plt.ylim((0, 1))


# animated plot objects
current_dAngle, = ax1.plot([],[],'ro', markersize=6)
current_angle, = ax2.plot([],[],'ro', markersize=6)

TSTEP = 0.0001
integral = 0.0
DECAY_RATE = 0.8
mu = 0
x = 0
counter = 0
for i in range(ids.size-2):
	counter += 1
	x += d_angles[i]
	mu += 1.0*(x-mu)/(counter)
	if counter > 5:
		integral += (x-mu)

	ax3_data.append(integral)
	ax3_plot.set_data(np.arange(i+1),ax3_data)

	current_angle.set_data(i, theta[i])
	current_dAngle.set_data(i, d_angles[i])

	if abs(integral) > 0.5 and counter > 5:
		mu = 0
		x = -d_angles[i+1]
		counter = 0
		integral = 0.0
		# PDF[:] = PDF_prior[:]
		ax1.plot(i, d_angles[i], 'r|', markersize=100)
		ax2.plot(i, theta[i], 'r|', markersize=100)

	# plt.draw()
	# plt.pause(TSTEP)

# plt.figure()
# plt.plot(x_coords, y_coords)


plt.show()
