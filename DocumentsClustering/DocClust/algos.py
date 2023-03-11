from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch
from sklearn_extra.cluster import CommonNNClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS


def kmeans(X, n_clusters, algorithm, init_centers):
    return KMeans(
        n_clusters = n_clusters, 
        init = init_centers, 
        algorithm = algorithm,
        random_state = 1, 
        n_init = 10, 
        tol = 1e-4
    ).fit(X).labels_


def kmedoids(X, n_clusters, metric, method, init_centers):
    return KMedoids(
        n_clusters = n_clusters,
        metric = metric,
        method = method, 
        init= init_centers, 
        max_iter = 3, 
        random_state = 1
    ).fit(X).labels_


def agglomerative(X, metric, dist_link_threshold, linkage):
    return AgglomerativeClustering(
        n_clusters = None, 
        metric = metric, 
        connectivity = None, 
        compute_full_tree = False, 
        linkage = linkage, 
        distance_threshold = dist_link_threshold, 
        compute_distances = False 
    ).fit(X).labels_


def birch(X, n_clusters, branching_factor, threshold):
    return Birch(
        branching_factor = branching_factor,
        n_clusters = n_clusters, 
        threshold = threshold
    ).fit(X).labels_


def dbscan(X, dist_nbr_threshold, metric_power):
    return DBSCAN(
        eps = dist_nbr_threshold, 
        min_samples = 5,
        algorithm = 'auto', 
        leaf_size = 30,
        p = metric_power 
    ).fit(X).labels_


def meanshift(X, bandwidth):
    return MeanShift(
        bandwidth = bandwidth
    ).fit(X).labels_


def optics(X, max_eps, metric):
	return OPTICS(
		min_samples = 5, 
		max_eps = max_eps, 
		metric = metric, 
		p = 1, 
		cluster_method = 'xi', 
		algorithm = 'auto', 
		leaf_size = 30,
	).fit(X).labels_


def common_nn(X, dist_nbr_threshold, metric):
	return CommonNNClustering(
		eps = dist_nbr_threshold, 
		min_samples = 5, 
		metric = metric, 
		algorithm = 'auto',
		leaf_size = 30, 
		p = 1 
	).fit(X).labels_