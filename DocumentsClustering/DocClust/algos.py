from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import Birch


def kmeans(X, n_clusters):

    return KMeans(
        n_clusters = n_clusters, 
        init ='k-means++', 
        random_state = 1, 
        n_init = 10, 
        tol = 1e-4
        ).fit(X)


def kmedoids(X, n_clusters):

    return KMedoids(
        n_clusters = n_clusters,
        metric = 'euclidean',
        method = 'alternate', 
        init= 'k-medoids++', 
        max_iter = 3, 
        random_state = 1
        ).fit(X)


def dbscan(X, dist_nbr_threshold):

    return DBSCAN(
        eps = dist_nbr_threshold, 
        min_samples = 2,
        algorithm = 'auto', 
        leaf_size = 30,
        p = 2, 
    ).fit(X)


def agglomerative_clst(X, n_clusters):

    return AgglomerativeClustering(
        n_clusters = n_clusters, 
        metric = "euclidean", 
        connectivity = None, 
        compute_full_tree = False, 
        linkage = "ward", 
        distance_threshold = None, 
        compute_distances = False 
    ).fit(X)


def agglomerative_dist(X, dist_link_threshold):

    return AgglomerativeClustering(
        n_clusters = None, 
        metric = "euclidean", 
        connectivity = None, 
        compute_full_tree = False, 
        linkage = "ward", 
        distance_threshold = dist_link_threshold, 
        compute_distances = False 
    ).fit(X)


def birch(X, branching_factor, threshold):

    return Birch(
        branching_factor = branching_factor,
        n_clusters = None, 
        threshold = threshold
    ).fit(X)



