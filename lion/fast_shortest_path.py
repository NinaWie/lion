from numba import jit
import numpy as np
from lion.utils.shortest_path import find, minus, plus


@jit(nopython=True)
def topological_sort_jit(v_x, v_y, shifts, to_visit, stack, curr_ind):
    """
    Recursive method for topological sorting of vertices in a DAG

    Arguments:
        v_x, v_y: current vertex / cell
        shifts: array of length n_neighborsx2 to iterate over neighbors
        to_visit: 2D array of size of instance to remember visited nodes
        stack: list of topologically sorted vertices
    Returns:
        stack of sorted cells
    """
    x_len, y_len = to_visit.shape
    # Mark the current node as visited.
    to_visit[v_x, v_y] = 0
    # Recur for all the vertices adjacent to this vertex
    for s in range(len(shifts)):
        neigh_x = v_x + shifts[s, 0]
        neigh_y = v_y + shifts[s, 1]
        # in array bounds and not visited yet
        if (
            neigh_x >= 0 and neigh_x < x_len and neigh_y >= 0
            and neigh_y < y_len and to_visit[neigh_x, neigh_y] == 1
        ):
            _, curr_ind = topological_sort_jit(
                neigh_x, neigh_y, shifts, to_visit, stack, curr_ind
            )
    # Push current vertex to stack which stores result
    stack[v_x, v_y] = curr_ind
    return stack, curr_ind + 1


@jit(nopython=True)
def del_after_dest(stack, d_x, d_y):
    # TODO: move to utils (not needed I think)
    """
    Helper method to use only the relevant part of the stack
    """
    for i in range(len(stack)):
        if stack[i, 0] == d_x and stack[i, 1] == d_y:
            return stack[i:]


@jit(nopython=True)
def edge_costs(
    stack, pos2node, shifts, edge_cost, instance, edge_inst, shift_lines,
    shift_costs, edge_weight
):
    """
    Pre-compute all costs on each edge from pylon and cable resistances

    Arguments:
        stack: np array of shape (n,2): order in which to consider the vertices
            MUST BE TOPOLOGICALLY SORTED for this algorithm to work
        pos2node: 2D ndarray with pos2node[x,y] = index of cell (x,y) in stack
        shifts: np array of size (x,2), indicating the neighborhood for each
            vertex
        edge_cost: 2Darray of size (n, len(shifts)), initially all inf
        instance: 2Darray of pylon resistance values for each cell
        edge_inst: 2Darray with resistances to traverse this cell with a cable
                   (often same as instance)
        shift_lines: numba typed List filled with len(shifts) np arrays,
                    each array of shape (x,2) is the Bresenham line connecting
                    a cell to one of its neighbors
        shift_costs: 1Darray of length len(shift) containing the Euclidean
                    length to each neighbor
        edge_weight: Weight defining importance of cable costs (=resistance to
                    traverse cell with a cable) compared ot pylon costs
    """
    edge_inst_len_x, edge_inst_len_y = edge_inst.shape
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]
        vertex_costs = instance[v_x, v_y]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < edge_inst_len_x
                and 0 <= neigh_y < edge_inst_len_y
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                bresenham_edge_dist = 0
                if edge_weight > 0:
                    bres_line = shift_lines[s] + np.array([v_x, v_y])
                    edge_cost_list = np.zeros(len(bres_line))
                    for k in range(1, len(bres_line) - 1):
                        edge_cost_list[k] = edge_inst[bres_line[k][0],
                                                      bres_line[k][1]]
                    # TODO: mean or sum?
                    bresenham_edge_dist = edge_weight * np.mean(edge_cost_list)
                neigh_ind = pos2node[neigh_x, neigh_y]
                edge_cost[neigh_ind, s] = shift_costs[s] * (
                    0.5 * (vertex_costs + instance[neigh_x, neigh_y]) +
                    bresenham_edge_dist
                )
    return edge_cost


@jit(nopython=True)
def update_default(angles_all, dists):
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
def sp_dag(stack, pos2node, shifts, angles_all, dists, preds, edge_cost):
    """
    Angle-weighted dynamic program for Directed Acyclic Graphs (O(n))
    Stores the distances and predecessors for all IN edges for each vertex

    Arguments:
        stack: np array of shape (n,2): order in which to consider the vertices
            MUST BE TOPOLOGICALLY SORTED for this algorithm to work
        pos2node: 2D ndarray with pos2node[x,y] = index of cell (x,y) in stack
        shifts: np array of size (x,2) --> indicating the neighborhood for each
            vertex
        angles_all: 2Darray of size (len(shifts), len(shifts))
                    precomputed angle cost for each tuple of edges
        dists: 2Darray of size (n, len(shifts)) - contains distance of each
               edge from the source vertex
        preds: 2Darray of size (n, len(shifts)) - contains optimal predecessor
               of each edge from the source vertex
        edge_cost: 2Darray of size (n, len(shifts)) - edge cost for each edge
    """

    inst_x_len, inst_y_len = pos2node.shape
    # print(len(stack))
    for i in range(len(dists)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]

        # get predecessor for all outgoing edges at this vertex
        # TODO: unnecessary updates for infeasible edges
        predecessors = update("linear", angles_all, dists[i])

        # get index and update
        for s in range(len(shifts)):
            neigh_x = int(v_x + shifts[s][0])
            neigh_y = int(v_y + shifts[s][1])
            if (
                0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                neigh_stack_ind = pos2node[neigh_x, neigh_y]
                # add up pylon cost + angle cost + edge cost
                pred = int(predecessors[s])
                cost_and_angle = dists[i, pred] + angles_all[s, pred]

                # update distances and predecessors
                dists[neigh_stack_ind,
                      s] = cost_and_angle + edge_cost[neigh_stack_ind, s]
                preds[neigh_stack_ind, s] = pred
    return dists, preds


@jit(nopython=True)
def update(func, angles_all, in_dists):
    """
    Interface to different update algorithms
    """
    if func == "linear":
        return update_linear(angles_all, in_dists)
    # TODO: use prepare_for_discrete from utils/shortest_path (but use it in
    # angle graph) and pass the arguments efficiently somehow
    # elif func == "discrete":
    #     return update_discrete(angles_all, in_dists)
    else:
        return update_default(angles_all, in_dists)


@jit(nopython=True)
def sp_dag_reversed(stack, pos2node, shifts, angles_all, dists, edge_cost):
    """
    Angle-weighted dynamic program for Directed Acyclic Graphs (O(n))
    Stores the distances and predecessors for all OUT edges for each vertex

    Arguments:
        stack: np array of shape (n,2): order in which to consider the vertices
            MUST BE TOPOLOGICALLY SORTED for this algorithm to work
        pos2node: 2D ndarray with pos2node[x,y] = index of cell (x,y) in stack
        shifts: np array of size (x,2) --> indicating the neighborhood for each
            vertex
        angles_all: 2Darray of size (len(shifts), len(shifts))
                    precomputed angle cost for each tuple of edges
        dists: 2Darray of size (n, len(shifts)) - contains distance of each
               edge from the source vertex
        preds: 2Darray of size (n, len(shifts)) - contains optimal predecessor
               of each edge from the source vertex
        edge_cost: 2Darray of size (n, len(shifts)) - edge cost for each edge
    """

    inst_len_x, inst_len_y = pos2node.shape
    n_neighbors = len(shifts)
    preds = np.zeros(dists.shape) - 1

    for i in range(len(stack)):
        v_x = stack[-i - 1, 0]
        v_y = stack[-i - 1, 1]

        # get in distances by computing the in neighbors
        in_inds = np.zeros(n_neighbors) - 1
        in_dists = np.zeros(n_neighbors) + np.inf
        for s_in in range(n_neighbors):
            in_neigh_x = v_x - shifts[s_in][0]
            in_neigh_y = v_y - shifts[s_in][1]
            if (
                0 <= in_neigh_x < inst_len_x and 0 <= in_neigh_y < inst_len_y
                and pos2node[in_neigh_x, in_neigh_y] >= 0
            ):
                in_neigh_ind = pos2node[in_neigh_x, in_neigh_y]
                in_inds[s_in] = int(in_neigh_ind)
                in_dists[s_in] = dists[in_neigh_ind, s_in]

        predecessors = update("linear", angles_all, in_dists)

        # update OUTGOING edge distances
        for s in range(n_neighbors):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            # out neighbor feasible?
            if (0 <= neigh_x <
                inst_len_x) and (0 <= neigh_y < inst_len_y
                                 ) and (pos2node[neigh_x, neigh_y] >= 0):

                # beginning: only update out edges of destination
                if i == 0:
                    dists[-1, s] = edge_cost[-i - 1, s]
                    continue
                pred = int(predecessors[s])
                cost_and_angle = in_dists[pred] + angles_all[s, pred]
                dists[-i - 1, s] = cost_and_angle + edge_cost[-i - 1, s]
                preds[-i - 1, s] = pred
    return dists, preds


@jit(nopython=True)
def sp_bf(
    n_iters, stack, shifts, angles_all, dists, preds, instance, edge_cost
):
    """
    Angle-weighted Bellman Ford algorithm for a general graph (not DAG)
    - stack does not need to be sorted
    Implemented with numba for performance - O(lm) where l is the
    maximum length of the shortest path

    Arguments:
        n_iters: Int - At most the number of vertices in the graph, if known
            then the maximum length of the shortest path
        stack: List of tuples - order in which to consider the vertices
            Note: For this algorithm it does not matter, because done for
            a sufficient number of iterations
        shifts: np array of size (x,2) --> indicating the neighborhood for each
            vertex
        angles_all: np array, angle cost for each shift (precomputed)
        dists: np array of size m --> indicates distance of each edge from the
            source vertex
        preds: np array of size m --> indicates predecessor for each edge
        instance: 2D array, for each vertex the cost
        edge_cost: np array of size m --> edge cost for each edge
    """
    for _ in range(n_iters):
        for i in range(len(stack)):
            v_x = stack[i][0]
            v_y = stack[i][1]
            for s in range(len(shifts)):
                neigh_x = v_x + shifts[s][0]
                neigh_y = v_y + shifts[s][1]
                if (
                    0 <= neigh_x < dists.shape[1]
                    and 0 <= neigh_y < dists.shape[2]
                    and instance[neigh_x, neigh_y] < np.inf
                ):
                    # add up pylon cost + angle cost + edge cost
                    cost_per_angle = dists[:, v_x, v_y] + angles_all[
                        s] + instance[neigh_x, neigh_y] + edge_cost[s, neigh_x,
                                                                    neigh_y]
                    # update distances and predecessors if better
                    if np.min(cost_per_angle) < dists[s, neigh_x, neigh_y]:
                        dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                        preds[s, neigh_x, neigh_y] = np.argmin(cost_per_angle)
    return dists, preds


@jit(nopython=True)
def update_linear(angles_all, dists):
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
def update_discrete(
    dists, alphas, tree_index, tree_values, discrete_costs, bounds
):
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
                min_angle_bound = minus(alphas[in_edge], bounds[dis_step, 1])
                max_angle_bound = minus(alphas[in_edge], bounds[dis_step, 0])
            else:
                min_angle_bound = plus(alphas[in_edge], bounds[dis_step, 0])
                max_angle_bound = plus(alphas[in_edge], bounds[dis_step, 1])
            # find the corresponding range of indices (binary search)
            fake_ind = np.arange(len(tree_values))
            min_angle_index = find(min_angle_bound, tree_values, fake_ind)
            if min_angle_index != 0 or min_angle_bound > tree_values[0]:
                min_angle_index += 1
            max_angle_index = find(max_angle_bound, tree_values, fake_ind) + 1
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
