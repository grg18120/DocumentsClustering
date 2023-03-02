from sklearn import metrics 

"""
External evaluation, clustering is compared to an existing "ground truth" classification
"""

def rand_index(labels_true, labels_pred):
	# Rand Index [0, 1] 
	return metrics.rand_score(labels_true, labels_pred)
		

def adjusted_rand_index(labels_true, labels_pred):
	# Adjusted Rand Index [-1, 1] 
	return metrics.adjusted_rand_score(labels_true, labels_pred)


def fowlkes_mallows_index(labels_true, labels_pred):
	"""
	Fowlkes-Mallows metric
	Values close to zero indicate two label assignments that are largely independent, 
	while values close to one indicate significant agreement. 
	Further, values of exactly 0 indicate purely independent label assignments and a FMI of exactly 1 indicates that 
	the two label assignments are equal 
	"""
	return metrics.fowlkes_mallows_score(labels_true, labels_pred)

def v_measure_index(labels_true, labels_pred):
	"""
	Homogeneity(h), completeness(c) and V-measure metric
	V-measure = [(1+b)*h*c] / [(b*h + c)]   b = 1.0(default)
	0.0 is as bad as it can be, 1.0 is a perfect score.
	h = metrics.homogeneity_score(labels_true, labels_pred)
	c = metrics.completeness_score(labels_true, labels_pred)
	v = ((1+b)*h*c) / ((b*h + c))   b = 1.0
	"""
	return metrics.v_measure_score(labels_true, labels_pred, beta = 1.0) 


def mutual_information(labels_true, labels_pred):
	return metrics.mutual_info_score(labels_true, labels_pred)


def adjusted_mutual_information(labels_true, labels_pred):
	return metrics.adjusted_mutual_info_score(labels_true, labels_pred)


def normalized_mutual_information(labels_true, labels_pred):
	return metrics.normalized_mutual_info_score(labels_true, labels_pred)


def pairwise_distances(cl_a, cl_b, metric):
    return metrics.pairwise_distances(cl_a, cl_b, metric = metric)