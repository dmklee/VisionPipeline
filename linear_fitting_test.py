import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt 
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib
from matplotlib.gridspec import GridSpec
plt.style.use('classic')


npr.seed(3)
LENGTH = 40
SLOPE = 0.
OFFSET = 0.

def linear_fit_error(x,y, model):
	return (model['A']*x + model['B']*y + model['C'])**2

def inc_linear_fitting(x,y,record={}):
	if len(record) == 0:
		record = {'sumX': 0.0,
				 'sumY': 0.0,
				 'sumXY': 0.0,
				 'sumX2': 0.0,
				 'length': 0}

	record['sumX'] += x
	record['sumY'] += y
	record['sumXY'] += x*y 
	record['sumX2'] += x*x 
	record['length'] += 1

	xMean = record['sumX'] / record['length']
	yMean = record['sumY'] / record['length']
	denom = record['sumX2'] - record['sumX']*xMean

	if abs(denom) < 1e-6:
		A = 1.0
		B = 0.0
		C = -xMean
	else:
		A = - (record['sumXY'] - record['sumX'] *yMean) / denom
		B = 1.0
		C = - (yMean + A * xMean)

	model = {'A' : A,
			 'B' : B,
			 'C' : C}

	return model, record

def calc_y(x, model):
	return -(model['A'] * x + model['C'])/ model['B']

def inc_follow_contour(X_, Y_, plot_objects):
	#unpack plot_objects
	rejects, accepts, linear_model, upper_lim, lower_lim, curv_text = plot_objects
	global record, model
	record = {}
	def helper(i):
		global record, model
		if i == 0:
			record = {}
		x = X_[i]
		y = Y_[i]
		error = 0.
		tol = max((0.5 * 0.85**i), 0.02)
		if i > 10:
			error = linear_fit_error(x, y, model) 
		if error < tol:
			accepts_x, accepts_y = accepts.get_data()
			accepts.set_data(list(accepts_x) + [x] , list(accepts_y)+[y])
			model, record = inc_linear_fitting(x, y, record)
		else:
			rejects_x, rejects_y = rejects.get_data()
			np.append(rejects_x,[x])
			np.append(rejects_y,[y])
			rejects.set_data(list(rejects_x) + [x] , list(rejects_y)+[y])
		if i > 2:
			x_lo = np.amin(X_[:(i+1)])
			x_hi = np.amax(X_[:(i+1)]) 
			linear_model.set_xdata([x_lo, x_hi])
			linear_model.set_ydata([calc_y(x_lo, model), 
									calc_y(x_hi, model)])
			upper_lim.set_xdata([x_lo, x_hi])
			upper_lim.set_ydata([calc_y(x_lo, model)+tol**0.5, 
									calc_y(x_hi, model)+tol**0.5])
			lower_lim.set_xdata([x_lo, x_hi])
			lower_lim.set_ydata([calc_y(x_lo, model)-tol**0.5, 
									calc_y(x_hi, model)-tol**0.5])
			r_est = abs(model['B']/model['A'])
			curv_text.set_text('Est. Radius: {} pixels'.format(np.round(r_est,1)))

		return rejects, accepts, linear_model, upper_lim, lower_lim, curv_text,

	return helper

# X_ = np.sort(100.*npr.uniform(size=LENGTH))
# Y_ = SLOPE * X_ + OFFSET + 0.3*(npr.random(size=LENGTH)-0.5)

IMG = "04"
DATA = np.loadtxt('data_' + IMG + '.txt', delimiter = ',')
X_ = DATA[:,0]
Y_ = DATA[:,1]

tstep = 10
DATA = DATA[DATA[:, 0].argsort()][:]
ids = DATA[:,0]
angle = DATA[:,1]
d_angle = DATA[:,2]
accum_d_angle = DATA[:,3]
corner_mask = DATA[:,4] == 1

def gaussian(x, mu, var):
	return np.exp(-np.power(x-mu, 2.)/(2.*var))

def step_corner_detection(ids, d_angle, plot_objects):
	#unpack objects
	current_datum, PDF, PDF_center, error_bar, corners, changes, PDF_datum = plot_objects
	global mu, S, counter, integral, state
	mu = 0.
	S = 0.02
	counter = 0
	integral = 0.
	state = 'walking' # 'on_corner'
	def helper(i):
		#update gaussian
		global mu, S, counter, integral, state
		integral *= 0.8
		delta = d_angle[i] - mu
		counter += 1
		conf95 = 2*np.sqrt(S/counter)
		if abs(delta) < conf95:
			mu += delta/counter
			delta2 = d_angle[i] - mu
			S += delta*delta2
			c = 'blue'
		else:
			integral += 0.1*np.sign(delta)
			c = 'red'
			counter -= 1
		#update plot objects
		current_datum.set_xdata(ids[i])
		current_datum.set_ydata(d_angle[i])

		PDF.set_ydata(gaussian(PDF.get_xdata(), mu, S/counter))
		PDF_center.set_xdata(2*[mu])
		PDF_datum.set_xdata(2*[d_angle[i]])
		PDF_datum.set_color(c)

		error_bar.set_ydata(integral)
		c = 'b' if integral > 0 else 'r'
		c = 'gray' if abs(integral) < 0.001 else c
		error_bar.set_color(c)
		if state == 'walking' and abs(integral) > 0.15:
			_x, _y = corners.get_data()
			corners.set_xdata(list(_x)+[ids[i]])
			corners.set_ydata(list(_y)+[0])
			state = 'on_corner'
			# walk back to find start of corner
			offset = 0
			for _ in range(20):
				offset -= 1
				if np.sign(integral)*d_angle[i+offset] < 0.01:
					break
			_x, _y = changes.get_data()
			changes.set_xdata(list(_x)+[ids[i+offset]])
			changes.set_ydata(list(_y)+[0])

		if state == 'on_corner' and np.sign(integral)*d_angle[i] < 0.01:
			mu = 0.
			S = 0.02
			counter = 0
			integral = 0.
			state = 'walking'
			_x, _y = changes.get_data()
			changes.set_xdata(list(_x)+[ids[i]])
			changes.set_ydata(list(_y)+[0])
		# if abs(integral) > 0.2:
		# 	mu = 0.
		# 	S = 0.02
		# 	counter = 0
		# 	integral = 0.
			
		return current_datum, PDF, PDF_center, error_bar, corners, changes, PDF_datum
	return helper


fig = plt.figure(figsize=(12,8))
plt.title('img_'+IMG)
gs1 = GridSpec(3,5)

# show angle
ax0 = fig.add_subplot(gs1[0,:-1])
ref_angle = np.cumsum(-d_angle)
ax0.plot(ids, ref_angle)
plt.ylim(np.amin(ref_angle)-0.3, np.amax(ref_angle)+0.3)
corners, = plt.plot([],[], 'r|', markersize=500)
changes, = plt.plot([],[], 'g|', markersize=500)



# shows d_angle
ax1 = fig.add_subplot(gs1[1,:-1])
ax1.plot(ids, d_angle, '-o', markersize=2)
current_datum, = plt.plot([],[], 'ro', markersize=4)

#shows guassian over slope
ax2 = fig.add_subplot(gs1[2,:-1])
plt.ylim((0,1.2))
plt.xlim((-0.3,0.3))
tmp = np.linspace(-0.3,0.3,1000)
PDF, = plt.plot(tmp, np.zeros_like(tmp))
PDF_center, = plt.plot([0,0],[0,1], 'k:')
PDF_datum, = plt.plot([0,0],[0,1.2], 'r', linewidth=4)
#shows error term
ax3 = fig.add_subplot(gs1[1:, -1])
plt.ylim((-0.3,0.3))
plt.xlim((-1,1))
plt.plot([-1,1],2*[-0.2], 'k--')
plt.plot([-1,1],2*[0.2], 'k--')
plt.plot([-1,1],2*[0], 'k')
error_bar, = plt.plot(0,0, 's', markersize=20)

plot_objects = (current_datum, PDF, PDF_center, error_bar, 
				corners, changes, PDF_datum)

anim = animation.FuncAnimation(fig, step_corner_detection(ids, d_angle, plot_objects),
								frames=ids.size, interval=tstep, blit=True, repeat=False)
plt.show()

# fig = plt.figure(figsize=(18,10))
# fig.add_subplot(311)
# plt.plot(ids, angle, '.-')
# plt.plot(ids[corner_mask], angle[corner_mask], 'ro')
# plt.xlim((np.amin(X_) - X_.size//8, np.amax(X_) + X_.size//8))
# plt.ylim((np.amin(Y_) - 0.2, np.amax(Y_)+0.2))

# fig.add_subplot(312)
# plt.plot(ids, d_angle, 'b.-')
# plt.plot(ids[corner_mask], d_angle[corner_mask], 'ro')
# plt.plot((np.amin(X_),np.amax(X_)), (0,0), 'k--')
# plt.xlim((np.amin(X_) - X_.size//8, np.amax(X_) + X_.size//8))

# fig.add_subplot(313)
# plt.plot(ids, np.abs(accum_d_angle))
# plt.xlim((np.amin(X_) - X_.size//8, np.amax(X_) + X_.size//8))
# plt.show()

# rejects, = plt.plot([],[],'r.', markersize=12)
# accepts, = plt.plot([],[],'g.', markersize=12)
# linear_model, = plt.plot([],[],'r--', linewidth=2)
# upper_lim, = plt.plot([],[],'k:')
# lower_lim, = plt.plot([],[],'k:')
# curv_text = plt.text(X_[-X_.size//6], 1.1*np.amax(Y_), '')

# plot_objects = (rejects, accepts, linear_model, upper_lim, lower_lim, curv_text)


# # anim = animation.FuncAnimation(fig, inc_follow_contour(X_, Y_, plot_objects), frames=X_.size-1, 
# 									# interval=10, blit=True, repeat=False)
# plt.show()