from numba import jit
import numpy as np


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
        for s in range(len(shifts)):
            neigh_x = int(v_x + shifts[s][0])
            neigh_y = int(v_y + shifts[s][1])
            if (
                0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                neigh_stack_ind = pos2node[neigh_x, neigh_y]
                # add up pylon cost + angle cost + edge cost
                cost_per_angle = dists[i] + angles_all[s]

                # update distances and predecessors
                dists[neigh_stack_ind,
                      s] = np.min(cost_per_angle) + edge_cost[neigh_stack_ind,
                                                              s]
                preds[neigh_stack_ind, s] = np.argmin(cost_per_angle)
    return dists, preds


@jit(nopython=True)
def average_lcp(stack, shifts, angles_all, dists, preds, instance, edge_cost):
    """
    Compute the least cost AVERAGE path (with running average)
    """
    counter = np.ones(dists.shape)
    # print(len(stack))
    for i in range(len(stack)):
        v_x = stack[i][0]
        v_y = stack[i][1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]
            if (
                0 <= neigh_x < dists.shape[1] and 0 <= neigh_y < dists.shape[2]
                and instance[neigh_x, neigh_y] < np.inf
            ):
                # add up pylon cost + angle cost + edge cost
                cost_per_angle = (
                    (dists[:, v_x, v_y] * counter[:, v_x, v_y]) +
                    angles_all[s] + instance[neigh_x, neigh_y] +
                    edge_cost[s, neigh_x, neigh_y]
                ) / (counter[:, v_x, v_y] + 1)  # div by counter + 1
                # update distances and predecessors
                dists[s, neigh_x, neigh_y] = np.min(cost_per_angle)
                preds[s, neigh_x, neigh_y] = np.argmin(cost_per_angle)
                # update counter
                counter[s, neigh_x,
                        neigh_y] = counter[np.argmin(cost_per_angle), v_x,
                                           v_y] + 1
    return dists, preds


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
    preds = np.zeros(dists.shape) - 1

    # update OUTGOING edge distances
    for i in range(len(stack)):
        v_x = stack[-i - 1, 0]
        v_y = stack[-i - 1, 1]
        for s in range(len(shifts)):
            neigh_x = v_x + shifts[s][0]
            neigh_y = v_y + shifts[s][1]

            if (0 <= neigh_x <
                inst_len_x) and (0 <= neigh_y < inst_len_y
                                 ) and (pos2node[neigh_x, neigh_y] >= 0):

                # beginning: only update out edges of destination
                if i == 0:
                    dists[-1, s] = edge_cost[-i - 1, s]
                    continue

                # iterate over incoming edges to find minimum
                min_angle_cost = np.inf
                min_angle_index = 0
                for s2 in range(len(shifts)):
                    in_neigh_x = v_x - shifts[s2][0]
                    in_neigh_y = v_y - shifts[s2][1]
                    if (
                        0 <= in_neigh_x < inst_len_x
                        and 0 <= in_neigh_y < inst_len_y
                        and pos2node[in_neigh_x, in_neigh_y] >= 0
                    ):
                        in_neigh_ind = pos2node[in_neigh_x, in_neigh_y]
                        curr_angle_cost = dists[in_neigh_ind,
                                                s2] + angles_all[s, s2]
                        if curr_angle_cost < min_angle_cost:
                            min_angle_cost = curr_angle_cost
                            min_angle_index = s2
                dists[-i - 1, s] = min_angle_cost + edge_cost[-i - 1, s]
                preds[-i - 1, s] = min_angle_index
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
def efficient_update_sp(
    stack, pos2node, shifts, angles_all, dists, preds, edge_cost
):
    """
    Implemented more efficient method for angle updates
    Corresponds to sp_dag, but with improved runtime (O(k log k + l) instead)
    of O(kl) for a vertex update with k incoming and l outgoing edges

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
    n_neighbors = len(shifts)

    # Iterate over vertices
    for i in range(len(dists)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]

        # sort the in edge distances and initialize
        initial_S = np.argsort(dists[i])
        marked_plus = np.zeros(n_neighbors)
        marked_minus = np.zeros(n_neighbors)

        # initialize dists and do first pass
        neighbor_inds = np.zeros(n_neighbors) - 1

        for s in range(n_neighbors):
            neigh_x = int(v_x + shifts[s][0])
            neigh_y = int(v_y + shifts[s][1])
            if (
                0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                neigh_stack_ind = pos2node[neigh_x, neigh_y]
                neighbor_inds[s] = neigh_stack_ind
                # initialize distances to the straight line value
                dists[neigh_stack_ind,
                      s] = dists[i, s] + edge_cost[neigh_stack_ind, s]
                preds[neigh_stack_ind, s] = s

        # set current tuple: in edge and shift
        # (out edge index unncessary because same as in edge)
        current_in_edge = initial_S[0]
        current_shift = 0
        tuple_counter = 0

        while tuple_counter < len(initial_S) - 1:
            # best out edge is exactly the same shift!
            current_out_edge = (current_in_edge + current_shift) % n_neighbors

            # compute possible update value:
            update_val = dists[i,
                               current_in_edge] + angles_all[current_out_edge,
                                                             current_in_edge]

            if current_shift == 0:
                marked = marked_plus[current_out_edge] and marked_minus[
                    current_out_edge]
            elif current_shift > 0:
                marked = marked_plus[current_out_edge]
            else:
                marked = marked_minus[current_out_edge]

            # update only if better
            neigh_stack_ind = int(neighbor_inds[current_out_edge])

            if marked == 0 and neigh_stack_ind >= 0 and np.around(
                update_val + edge_cost[neigh_stack_ind, current_out_edge], 5
            ) <= np.around(dists[neigh_stack_ind, current_out_edge], 5):
                dists[neigh_stack_ind,
                      current_out_edge] = update_val + edge_cost[
                          neigh_stack_ind, current_out_edge]
                preds[neigh_stack_ind, current_out_edge] = current_in_edge

                # progress one edge further
                progress_one = True

            # inf neighbor --> jump over it if its incoming edge is worse
            elif marked == 0 and neigh_stack_ind < 0 and np.around(
                update_val, 5
            ) <= np.around(dists[i, current_out_edge], 5):
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

    return dists, preds
