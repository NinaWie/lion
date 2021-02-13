from numba import jit
import numpy as np
from lion.utils.general import angle, angle_360


def get_update_algorithm(func, angle_cost_array, in_edges, out_edges=None):
    """
    Interface to update algorithms - dependent on the angle cost function,
    choose the appropriate update algorithm and return it with the necessary
    arguments

    Arguments:
        func: str defining the angle cost function
        angle_cost_array: n x n 2D numpy array containing the costs for each
            combination of vectors
        in_edges: n x 2 shaped array with vectors of incoming edges
        out_edges: n x 2 shaped array with vectors of outgoing edges
    Returns:
        algorithm: function, update algorithm to be used
        args: Tuple, corresponding arguments
    """
    if func == "linear" and len(angle_cost_array) > 50 and np.all(
        angle_cost_array < np.inf
    ):
        # >50 because hidden factors in O-notation - will be slower
        # all must be smaller inf becasue otherwise it's nonlinear
        args = (angle_cost_array)
        algorithm = update_linear
    elif func == "discrete" and len(angle_cost_array) > 100:
        # >100 because hidden factors in O-notation - will be slower
        args = prepare_for_discrete(angle_cost_array, in_edges, out_edges)
        algorithm = update_discrete
    else:
        args = (angle_cost_array)
        algorithm = update_default
    return algorithm, args


def prepare_for_discrete(angle_cost_array, in_edges, out_edges=None):
    """
    Computation of special arguments for update algorithm for discrete angle
    cost functions
    Arguments:
        angle_cost_array: n x n 2D numpy array containing the costs for each
            combination of vectors
        in_edges: n x 2 shaped array with vectors of incoming edges
        out_edges: n x 2 shaped array with vectors of outgoing edges
    Returns:
        alphas: Angles of incoming edges with respect to reference vector
        tree_index: 1D array, sorted indices of outgoing angles
        tree_values: 1D array, sorted values of outgoing angles
        discrete_costs: 1D array of length d, possible angle costs (discrete
                function, so should only be a few)
        bounds: 2D array of shape d x 2, minimum and maximum angle for each
                possible angle cost
    """
    # normalize in and out edges:
    in_edges_norm = [np.asarray(s) for s in in_edges]
    in_edges_norm = [s / np.linalg.norm(s) for s in in_edges_norm]
    if out_edges is not None:
        out_edges_norm = [np.asarray(s) for s in out_edges]
        out_edges_norm = [s / np.linalg.norm(s) for s in out_edges_norm]
    else:
        out_edges_norm = in_edges_norm

    # get unique discrete costs and bounds
    discrete_costs = np.unique(angle_cost_array)
    d_uni = len(discrete_costs)

    # get bounds
    angles_raw = np.array(
        [
            [
                angle(in_edge, out_edge, normalize=False)
                for in_edge in in_edges_norm
            ] for out_edge in out_edges_norm
        ]
    )
    bounds = np.zeros((d_uni, 2))
    prev_max = 0
    for i, dis_cost in enumerate(discrete_costs):
        covered_angles = angles_raw[angle_cost_array == dis_cost]
        in_between = 0.5 * (prev_max + np.min(covered_angles))
        if i > 0:
            bounds[i - 1, 1] = in_between
        prev_max = np.max(covered_angles)
        bounds[i, 0] = in_between
    bounds[i, 1] = prev_max + 1

    # compute alphas and betas
    ref_edge = [1, 0]
    alphas = np.asarray(
        [
            angle_360(e, ref_edge, normalize=False) + np.pi
            for e in in_edges_norm
        ]
    )
    betas = np.array(
        [
            angle_360(e, ref_edge, normalize=False) + np.pi
            for e in out_edges_norm
        ]
    )

    tree_index = np.argsort(betas)
    tree_values = betas[tree_index]

    return alphas, tree_index, tree_values, discrete_costs, bounds


@jit(nopython=True)
def update_default(dists, angles_all):
    """
    Trivial updates working for any angle cost function
    O(kl) for k in edges and l out edges
    """
    predecessors = np.zeros(len(angles_all), dtype=np.int64)
    for s in range(len(angles_all)):
        in_costs = dists + angles_all[s]
        predecessors[s] = np.argmin(in_costs)
    return predecessors


@jit(nopython=True)
def update_linear(dists, angles_all):
    """
    Efficient update mechanism for LINEAR angle cost functions, i.e. the angle
    cost increases linearly with the angle
    Improves the runtime compared to update_default to O((k+l) log kl) instead
    of O(kl) for a vertex update with k incoming and l outgoing edges

    Arguments:
        angles_all: 2Darray of size (len(shifts), len(shifts))
                    precomputed angle cost for each tuple of edges
        dists: 1D array of size (len(shifts)) which are the distances to the
                    incoming edges
    Returns:
        1D int array containing the optimal predecessor for each out edge
    """
    n_neighbors = len(angles_all)

    # sort the in edge distances and initialize
    initial_S = np.argsort(dists)
    marked_plus = np.zeros(n_neighbors)
    marked_minus = np.zeros(n_neighbors)

    # for now, assign all edges its straight line predecessor
    predecessors = np.arange(n_neighbors)
    preliminary_dist = np.zeros(n_neighbors)
    for s in range(n_neighbors):
        preliminary_dist[s] = dists[s] + angles_all[s, s]

        # set current tuple: in edge and shift
    # (out edge index unncessary because same as in edge)
    current_in_edge = initial_S[0]
    current_shift = 0
    tuple_counter = 0

    while tuple_counter < len(initial_S) - 1:
        # best out edge is exactly the same shift!
        current_out_edge = (current_in_edge + current_shift) % n_neighbors

        # compute possible update value:
        update_val = dists[current_in_edge] + angles_all[current_out_edge,
                                                         current_in_edge]

        if current_shift == 0:
            marked = marked_plus[current_out_edge] and marked_minus[
                current_out_edge]
        elif current_shift > 0:
            marked = marked_plus[current_out_edge]
        else:
            marked = marked_minus[current_out_edge]

        # update only if better
        if marked == 0 and np.around(update_val, 5) <= np.around(
            preliminary_dist[current_out_edge], 5
        ):
            preliminary_dist[current_out_edge] = update_val
            predecessors[current_out_edge] = int(current_in_edge)

            # progress one edge further
            progress_one = True

        # already marked or update not successful:
        # Consider first edge in other direction or next overall tuple
        else:
            progress_one = False
            if current_shift > 0:
                current_shift = -1
            else:
                # get next tuple from stack
                tuple_counter += 1
                current_in_edge = initial_S[tuple_counter]
                current_shift = 0

        # Progress to next edge
        if progress_one:
            if current_shift < 0:
                current_shift -= 1
            if current_shift <= 0:
                marked_minus[current_out_edge] = 1
            if current_shift >= 0:
                current_shift += 1
                marked_plus[current_out_edge] = 1

    return predecessors


@jit(nopython=True)
def _circular_plus(a, b):
    """
    Plus op on circular x-axis: If sum is greater than 2pi, rotate to beginning
    Arguments:
        a,b: floats between 0 and 2 * np.pi (angle values)
    """
    return (a + b) % (2 * np.pi)


@jit(nopython=True)
def _binary_index_search(key, tree_values, tree_index):
    """
    Binary search to find the closest (lower) value to <key> in an array
    (recursive method)

    Arguments:
        key: Float, the value to look for
        tree_values: 1D sorted array of values in the tree
        tree_index: 1D int array of same length as tree_values, contains the
                original indices for the sorted value of tree_values
    Returns:
        The index in tree_index where the corresponding value in tree_values
        is closest to key (closest from left side, i.e. the lower one)
    """
    tree_len = len(tree_values)
    if tree_len == 1:
        return tree_index[0]
    middle = tree_len // 2
    if key < tree_values[middle]:
        return _binary_index_search(
            key, tree_values[:middle], tree_index[:middle]
        )
    else:
        return _binary_index_search(
            key, tree_values[middle:], tree_index[middle:]
        )


@jit(nopython=True)
def _circular_minus(a, b):
    """
    Minus op on circular x-axis - if smaller zero, then subtract from 2 pi
    Arguments:
        a,b: floats between 0 and 2 * np.pi (angle values)
    """
    if a < b:
        return 2 * np.pi - b + a
    else:
        return a - b


@jit(nopython=True)
def update_discrete(dists, args):
    """
    Efficient update mechanism for DISCRETE angle cost functions, i.e. there
    are only d possible angle costs
    Improves the runtime compared to update_default to O(dk log l) instead
    of O(kl) for a vertex update with k incoming and l outgoing edges

    Arguments:
        angles_all: 2Darray of size (len(shifts), len(shifts))
                    precomputed angle cost for each tuple of edges
        dists: 1D array of size (len(shifts)) which are the distances to the
                    incoming edges
    Returns:
        1D int array containing the optimal predecessor for each out edge
    """
    (alphas, tree_index, tree_values, discrete_costs, bounds) = args
    # sort tuples of dists and possible angle costs
    d_uni = len(discrete_costs)
    n_in_edges = len(alphas)

    dists_tuples = np.zeros(n_in_edges * d_uni)
    for i in range(n_in_edges):
        for d in range(d_uni):
            dists_tuples[d * n_in_edges + i] = dists[i] + discrete_costs[d]
    sorted_inds = np.argsort(dists_tuples)
    sorted_tuples = np.zeros((len(sorted_inds), 2), dtype=np.int64)
    for s in range(len(sorted_inds)):
        ind = sorted_inds[s]
        sorted_tuples[s, 0] = ind % n_in_edges
        sorted_tuples[s, 1] = ind // n_in_edges

    predecessor = np.zeros(len(tree_values)) - 1
    is_updated = 0

    for s in range(len(sorted_tuples)):
        in_edge = sorted_tuples[s, 0]
        dis_step = sorted_tuples[s, 1]
        # in edge is the index of the incoming edge --> get corresponding alpha
        for j in range(2):
            # compute the current range of betas
            if j == 0:
                min_angle_bound = _circular_minus(
                    alphas[in_edge], bounds[dis_step, 1]
                )
                max_angle_bound = _circular_minus(
                    alphas[in_edge], bounds[dis_step, 0]
                )
            else:
                min_angle_bound = _circular_plus(
                    alphas[in_edge], bounds[dis_step, 0]
                )
                max_angle_bound = _circular_plus(
                    alphas[in_edge], bounds[dis_step, 1]
                )
            # find the corresponding range of indices (binary search)
            fake_ind = np.arange(len(tree_values))
            min_angle_index = _binary_index_search(
                min_angle_bound, tree_values, fake_ind
            )
            if min_angle_index != 0 or min_angle_bound > tree_values[0]:
                min_angle_index += 1
            max_angle_index = _binary_index_search(
                max_angle_bound, tree_values, fake_ind
            ) + 1
            if max_angle_bound < min_angle_bound:
                inds_inbetween = np.concatenate(
                    (
                        tree_index[min_angle_index:],
                        tree_index[:max_angle_index]
                    )
                )
            else:
                inds_inbetween = tree_index[min_angle_index:max_angle_index]
            # update predecessor
            for index in inds_inbetween:
                if predecessor[index] < 0:
                    predecessor[index] = in_edge
                    is_updated += 1
        # TODO: experiment whether it's worth to make the tree smaller
        if is_updated == len(tree_values):
            break
    return predecessor
