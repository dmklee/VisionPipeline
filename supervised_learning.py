

def read_data():
	f = open("labeled_corners_thetas.txt","r")
	contents = f.read()

	THETAS = []
	CORNERS = []
	for i, line in enumerate(contents.splitlines()):
		data = line.split(")")[0]
		data = [float(a) for a in data[1:].split(",")]
		THETAS.append(np.array(data))
		
		offset = 0
		kept = ""
		tuples = []
		current = ""
		for c in reversed(line):
			if c == ')': 
				offset += 1
			elif c == '(': 
				offset -= 1
				if len(current) > 0:
					tuples.append(current[::-1])
					current = ""
			elif offset == 2:	
				current += c
			kept += c
			if offset == 0:
				break
		labels = []
		for t in tuples[::-1]:
			z = t.split(',')
			labels.extend(range(int(z[0]),int(z[2])+1))
		bit_labels = np.zeros(len(data))
		bit_labels[labels] = 1
		CORNERS.append(bit_labels)

	for i in range(len(THETAS)):
		data = THETAS[i]
		bit_label = CORNERS[i]
		length = data.size
		THETAS.append(data[::-1])
		CORNERS.append(bit_labels[::-1])

	return THETAS, CORNERS

def bitwise_error(truth, est):
	return np.sum(truth != est)

def weighted_error(truth, est, weight_fp, weight_fn):
	# weight_fp is the penalty for labeling a non-corner as a corner
	# weight_np is the penalty for not identifying a true corner
	ret = truth - est
	ret[ret == -1] = weight_fp
	ret[ret == 1] = weight_fn
	return np.sum(ret)

def dist_nearest_error(truth, est):
	#assumes both are sorted in increasing order
	pass



def linear_fit(data,  error_threshold, decay_rate):
	sumX = 0
	sumY = 0
	summXY = 0
	sumX2 = 0
	n = 0
	for x,y in data:
		n += 1
		sumX += x
		sumY += y
		sumXY += x*y
		sumX2 += x*x
		xMean = sumX/n
		yMean = sumY/n
		denom = sumX2 - sumX*xMean
		if abs(denom) < 1e-6:
			A = 1
			B = 0
			C = -xMean
		else:
			A = - (sumXY - sumX*yMean)/denom
			B = 1
			C = - (yMean + A*xMean)
	


for i in range(0,20,3):
	X,Y = read_data()
	visualize_labeling(X[i], Y[i], show=False)
plt.show()
	
