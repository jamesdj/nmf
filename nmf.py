import itertools

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.cluster.hierarchy import cophenet, average
import scipy.spatial.distance as ssd
import matplotlib as mpl
import matplotlib.pyplot as plt


def select_nmf_k_by_silhouette(scaled, min_k=2, max_k=None, n_reps=3, outfile=None):
    if max_k is None:
        max_k = int(min(scaled.shape) / 2)
    from sklearn.metrics.cluster import silhouette_score
    silhouettes = []
    k_range = range(min_k, max_k)
    for n_latent in k_range:
        ss_ = []
        for _ in range(n_reps):
            nmf = NMF(init='nndsvd', n_components=n_latent, shuffle=True)
            nmf.fit(scaled)
            cluster_assignments = discretize_clustering(nmf, x=scaled)
            try:
                ss = silhouette_score(scaled, cluster_assignments) # I may need a more appropriate distance metric?
            except ValueError as e:
                print(e)
                ss = -1
            ss_.append(ss)
        silhouettes.append(np.mean(ss_))
    kr = np.array(range(min_k, len(silhouettes) + min_k))
    selected_k = kr[np.argmax(silhouettes)]
    plt.plot(range(3, len(silhouettes) + 3), silhouettes)
    plt.xlabel('number of components', fontsize=20)
    plt.ylabel('silhouette score', fontsize=20)
    plt.tick_params(labelsize=15)
    fig = mpl.pyplot.gcf()
    fig.set_size_inches(10, 10)
    plt.show()
    if outfile is not None:
        plt.savefig(outfile)
    return selected_k

def condense_symmetric_matrix(m):
    # convert the redundant n*n square matrix form into a condensed nC2 array
    # distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
    return ssd.squareform(m)

def cophenetic_correlation(distance_matrix):
    condensed = condense_symmetric_matrix(distance_matrix)
    #linkage = ward(distance_matrix)
    linkage = average(condensed) # would a different linkage function work better?
    c, d = cophenet(linkage, condensed)
    return c

def discretize_clustering(nmf, x=None, axis='samples'):
    """
    This is correct standard practice, though there may be a slightly superior way, finding a discrete solution
    closest to the continuous solution using some metric. E.g. Yu et al. 2003 Multiclass spectral clustering
    :param nmf:
    :param x:
    :param axis:
    :return:
    """
    if axis == 'features':
        soft_clustering = nmf.components_
        argmax_axis = 0
    elif axis == 'samples':
        if x is None:
            raise ValueError("x must be passed if clustering samples rather than features")
        soft_clustering = nmf.transform(x)
        argmax_axis = 1
    else:
        raise ValueError("Axis argument must be one of ['samples', 'features']")
    return np.argmax(soft_clustering, axis=argmax_axis)
# Todo: would it make more sense to return the index of the component in which they have the highest rank?
# rather than the component in which they have the highest coefficient?

def compute_consensus_distance_matrix(nmfs, x=None, axis='samples', permute=False):
    # Idea: they should coincide more for smaller k, just because there are fewer classes to choose from
    # so I should normalize the cophenetic coefficient somehow to some random background?
    # maybe permute the clusterings to get a different distance matrix and coph corr, divide?
    n, m = x.shape
    if axis == 'features':
        dim = m
    elif axis == 'samples':
        dim = n
    n_nmfs = float(len(nmfs))
    distance_matrix = np.eye(dim) * n_nmfs
    for nmf in nmfs:
        clustering = discretize_clustering(nmf, x=x, axis=axis)
        if permute:
            clustering = np.random.permutation(clustering)
        n_to_cluster = len(clustering)
        for i in range(n_to_cluster):
            for j in range(i + 1, n_to_cluster):
                if clustering[i] == clustering[j]:
                    distance_matrix[i, j] += 1
                    distance_matrix[j, i] += 1
    distance_matrix /= n_nmfs
    distance_matrix = 1.0 - distance_matrix
    return distance_matrix


def select_k_by_cophenetic_corr():
    pass


def align_bases(w1, w2, return_dist=False):
    # Todo: Allow other axis. If X.shape == (N, M) and M > N, would it be better to align H than W?
    # H would contain more data to align, may be more robust...?
    k = w1.shape[1]
    order = range(k)
    min_mse = 10E20
    best_order = order
    if k < 5:
        for permutation in itertools.permutations(order): # there are faster ways but it's not a bottleneck
            w2_perm = w2[:, permutation]
            mse = mean_squared_error(w1, w2_perm)
            if mse < min_mse:
                min_mse = mse
                best_order = permutation
    else:
        pairwise_col_dists = np.ones((k, k)) * -1
        idxs_left = list(range(k))
        greedy_best_order = []
        for i in range(k):
            col_i = w1[:,i]
            dists = np.array([mean_squared_error(col_i, w2[:,j]) for j in idxs_left])
            min_idx = np.argmin(dists)
            idx = idxs_left[min_idx]
            greedy_best_order.append(idx)
            idxs_left.remove(idx)
        best_order = greedy_best_order
        min_mse = mean_squared_error(w1, w2[:, best_order])
    if return_dist:
        return best_order, min_mse
    return best_order