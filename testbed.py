import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('classic')

def gaussian(x, mu, var):
	return np.exp(-np.power(x-mu, 2.)/(2.*var))

DATA = np.loadtxt('data.txt', delimiter = ',')[:-5]
# DATA = DATA[DATA[:, 0].argsort()]

# parameters
FRONT_LENGTH = 5
ERROR_TOL = 0.15
ERROR_ALPHA = 1./(FRONT_LENGTH)

MU_FRONT = []
MU_LAG = []
ERROR = []
ERROR_SMOOTHED = []
CORNERS = []
END_CORNERS = []

total_i = 0
i = 0
error = 0.0
mu_front = 0.0
mu_lag = 0.0
archive = np.zeros(FRONT_LENGTH)
offset = 0
x = 0
mu_error = 0.0
onCorner = False
for del_grad_id in DATA[:,0]:
	total_i += 1
	i += 1

	x = x+del_grad_id if i != 0 else 0
	x_lag = archive[0]
	#update archive
	for j in range(FRONT_LENGTH-1):
		archive[j] = archive[j+1]
	archive[FRONT_LENGTH-1] = x

	delta_front = x - mu_front
	mu_front = np.mean(archive)
	MU_FRONT.append(mu_front)

	if i > FRONT_LENGTH:
		delta_lag = (x_lag-mu_lag)
		mu_lag += delta_lag/(i+1-FRONT_LENGTH)
	else:
		mu_lag = 0*mu_front
	MU_LAG.append(mu_lag)

	if i > 2*FRONT_LENGTH:
		error = mu_front - mu_lag
		delta_error = error -mu_error
		mu_error += (error -mu_error)*ERROR_ALPHA
	ERROR.append(error)
	ERROR_SMOOTHED.append(mu_error)
	if onCorner:
		if np.sign(delta_error) != corner_side:
			
			i = 0
			x = 0
			error = 0.0
			mu_error = 0.0
			mu_front = 0.0
			mu_lag = 0.0
			archive[:] = 0
			onCorner = False
			END_CORNERS.append(total_i)
	elif abs(mu_error) > ERROR_TOL:
		CORNERS.append(total_i-FRONT_LENGTH)
		corner_side = np.sign(mu_error)
		onCorner = True

		
		
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax1.plot(DATA[:,1], 'o-', markersize=2)
ax1.set_ylabel('direction')
ax1.set_ylim((np.amin(DATA[:,1])-2, np.amax(DATA[:,1])+2))
ax1.set_xlim((0, (DATA.shape[0]//50+1)*50))
ax1.plot(MU_FRONT)
ax1.plot(MU_LAG)
ax1.plot(CORNERS, len(CORNERS)*[0], 'r|', markersize=300)
ax1.legend(["Data", "Frontier Fit", "Lag Fit"], loc=2, fontsize=8)

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot([0,1000], 2*[0], '--',color='gray')
ax2.plot([0,1000], 2*[ERROR_TOL], 'k:')
ax2.plot([0,1000], 2*[-ERROR_TOL], 'k:')
ax2.set_xlim((0, (DATA.shape[0]//50+1)*50))
ax2.set_ylim((-2.5*ERROR_TOL, 2.5*ERROR_TOL))
ax2.plot(ERROR, alpha=0.5)
ax2.plot(ERROR_SMOOTHED)

plt.figure()
plt.plot(DATA[:,2], DATA[:,3])
plt.plot(DATA[CORNERS,2], DATA[CORNERS,3], 'rs', markersize=3)
plt.plot(DATA[END_CORNERS,2], DATA[END_CORNERS,3], 'r^', markersize=3)
plt.plot(DATA[0,2], DATA[0,3], 'go', markersize=5)
plt.axis('equal')
plt.show()
