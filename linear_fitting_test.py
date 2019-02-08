import numpy as np 
import numpy.random as npr 
import matplotlib.pyplot as plt 
from matplotlib import animation
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
		tol = max((0.05 * 0.85**i), 0.002)
		if i > 1:
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
			linear_model.set_xdata([X_[0], X_[i]])
			linear_model.set_ydata([calc_y(X_[0], model), 
									calc_y(X_[i], model)])
			upper_lim.set_xdata([X_[0], X_[i]])
			upper_lim.set_ydata([calc_y(X_[0], model)+tol**0.5, 
									calc_y(X_[i], model)+tol**0.5])
			lower_lim.set_xdata([X_[0], X_[i]])
			lower_lim.set_ydata([calc_y(X_[0], model)-tol**0.5, 
									calc_y(X_[i], model)-tol**0.5])
			r_est = abs(model['B']/model['A'])
			curv_text.set_text('Est. Radius: {} pixels'.format(np.round(r_est,1)))

		return rejects, accepts, linear_model, upper_lim, lower_lim, curv_text,

	return helper

# X_ = np.sort(100.*npr.uniform(size=LENGTH))
# Y_ = SLOPE * X_ + OFFSET + 0.3*(npr.random(size=LENGTH)-0.5)

Y_ = (np.loadtxt('data.txt', delimiter = ',')[1:])[::-1]
delta = Y_[1:] - Y_[:-1]
if (np.abs(delta) > 1).any():
	i = np.argmax(np.abs(delta))
	Y_[i+1:] -= delta[i]
X_ = np.arange(Y_.size)



fig = plt.figure(figsize=(18,10))
plt.plot(X_,Y_, '.-')
plt.xlim((X_[0] - X_.size//8, X_[-1] + X_.size//8))
plt.ylim((0.8*np.amin(Y_), 1.2*np.amax(Y_)))

# plt.show()

rejects, = plt.plot([],[],'r.', markersize=12)
accepts, = plt.plot([],[],'g.', markersize=12)
linear_model, = plt.plot([],[],'r--', linewidth=2)
upper_lim, = plt.plot([],[],'k:')
lower_lim, = plt.plot([],[],'k:')
curv_text = plt.text(X_[-X_.size//6], 1.1*np.amax(Y_), '')

plot_objects = (rejects, accepts, linear_model, upper_lim, lower_lim, curv_text)


anim = animation.FuncAnimation(fig, inc_follow_contour(X_, Y_, plot_objects), frames=X_.size, 
									interval=30, blit=True, repeat=False)
plt.show()
