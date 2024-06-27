import numpy as np
import tensorflow as tf
from scipy.spatial import distance_matrix

#from example import som, input_data


def get_umatrix_optimized(som, input_data, weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights and bmu indices of all the input data points

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    NEED TO UDPATE FOR NEW VERSION
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray`
    :return: input_size x 1 bmu indices 'ndarray'
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))

    # iterate through each unit and find its neighbours on the map
    for j in range(m):
        for i in range(n):
            cneighbor_idxs = list()

            # Save the neighbours for a unit with location i, j
            if i > 0:
                cneighbor_idxs.append(j * n + i - 1)
            if i < n - 1:
                cneighbor_idxs.append(j * n + i + 1)
            if j > 0:
                cneighbor_idxs.append(j * n + i - n)
            if j < m - 1:
                cneighbor_idxs.append(j * n + i + n)

            # Get the weights of the neighbouring units
            cneighbor_weights = weights[cneighbor_idxs]

            # Get the average distance between unit i, j and all of its neighbors
            # Expand dims to broadcast to each of the neighbors
            umatrix[j * n + i] = distance_matrix(np.expand_dims(weights[j * n + i], 0), cneighbor_weights).mean()

    bmu_indices = som.bmu_indices(tf.constant(input_data, dtype=tf.float32))

    return umatrix, bmu_indices


def get_umatrix(input_vects, weights, m, n):
    """ Generates an n x m u-matrix of the SOM's weights and bmu indices of all the input data points

    Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
    When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
    encode similar information.
    :param weights: SOM weight matrix, `ndarray`
    :param m: Rows of neurons
    :param n: Columns of neurons
    :return: m x n u-matrix `ndarray`
    :return: input_size x 1 bmu indices 'ndarray'
    """
    umatrix = np.zeros((m * n, 1))
    # Get the location of the neurons on the map to figure out their neighbors. I know I already have this in the
    # SOM code but I put it here too to make it easier to follow.
    neuron_locs = list()
    for i in range(m):
        for j in range(n):
            neuron_locs.append(np.array([i, j]))
    # Get the map distance between each neuron (i.e. not the weight distance).
    neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

    for i in range(m * n):
        # Get the indices of the units which neighbor i
        neighbor_idxs = neuron_distmat[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
        # Get the weights of those units
        neighbor_weights = weights[neighbor_idxs]
        # Get the average distance between unit i and all of its neighbors
        # Expand dims to broadcast to each of the neighbors
        umatrix[i] = distance_matrix(np.expand_dims(weights[i], 0), neighbor_weights).mean()

    bmu_indices = []
    for vect in input_vects:
        min_index = min([i for i in range(len(list(weights)))],
                        key=lambda x: np.linalg.norm(vect -
                                                     list(weights)[x]))
        bmu_indices.append(neuron_locs[min_index])

    return umatrix, bmu_indices
