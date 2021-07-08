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
def edge_costs(
    stack, pos2node, shifts, edge_cost, instance, edge_inst, shift_lines,
    shift_costs, edge_weight
):
    """
    Pre-compute all costs on each edge from resistances at the points or if
    edge_weight>0 then also including resistances between points

    Arguments:
        stack: np array of shape (n,2): order in which to consider the vertices
            MUST BE TOPOLOGICALLY SORTED for this algorithm to work
        pos2node: 2D ndarray with pos2node[x,y] = index of cell (x,y) in stack
        shifts: np array of size (x,2), indicating the neighborhood for each
            vertex
        edge_cost: 2Darray of size (n, len(shifts)), initially all inf
        instance: 2Darray of point resistance values for each cell
        edge_inst: 2Darray with resistances to traverse this cell
                   (often same as instance)
        shift_lines: numba typed List filled with len(shifts) np arrays,
                    each array of shape (x,2) is the Bresenham line connecting
                    a cell to one of its neighbors
        shift_costs: 1Darray of length len(shift) containing the Euclidean
                    length to each neighbor
        edge_weight: Weight defining importance of costs between points
                compared to costs at the points themselves
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
                # compute resistances along bresenham line (straight line
                # through raster, corresponding to cable)
                bresenham_edge_dist = 0
                if edge_weight > 0:
                    bres_line = shift_lines[s] + np.array([v_x, v_y])
                    edge_cost_list = np.zeros(len(bres_line) - 2)
                    # sum up the resistances of all crossed cells
                    for k in range(1, len(bres_line) - 1):
                        edge_cost_list[k - 1] = edge_inst[bres_line[k][0],
                                                          bres_line[k][1]]
                    # edge cost = average cable resistance x length of cable
                    # --> If edge_weight = 1, cable resistances are as costly
                    # as pylon resistances
                    bresenham_edge_dist = (
                        shift_costs[s] * edge_weight * np.mean(edge_cost_list)
                    )
                # get index of neighboring pylon that is reached via shift s
                neigh_ind = pos2node[neigh_x, neigh_y]
                # raw pylon costs
                pylon_cost = 0.5 * (vertex_costs + instance[neigh_x, neigh_y])
                # geometric costs (indicated if edge weight negative)
                if edge_weight < 0:
                    bresenham_edge_dist = pylon_cost * (shift_costs[s] - 1)

                # sum up resistances for only the pylons
                edge_cost[neigh_ind, s] = pylon_cost + bresenham_edge_dist
    return edge_cost


@jit(nopython=True)
def sp_dag(
    stack, pos2node, shifts, angles_all, dists, edge_cost, algorithm, *args
):
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
        edge_cost: 2Darray of size (n, len(shifts)) - edge cost for each edge
    """
    inst_x_len, inst_y_len = pos2node.shape
    preds = np.zeros(dists.shape) - 1
    # print(len(stack))
    for i in range(len(dists)):
        v_x = stack[i, 0]
        v_y = stack[i, 1]

        # get predecessor for all outgoing edges at this vertex
        predecessors = algorithm(dists[i], *args)

        # get index and update
        for s in range(len(shifts)):
            neigh_x = int(v_x + shifts[s][0])
            neigh_y = int(v_y + shifts[s][1])
            if (
                0 <= neigh_x < inst_x_len and 0 <= neigh_y < inst_y_len
                and pos2node[neigh_x, neigh_y] >= 0
            ):
                neigh_stack_ind = pos2node[neigh_x, neigh_y]
                # add up point cost + angle cost + edge cost
                pred = int(predecessors[s])
                cost_and_angle = dists[i, pred] + angles_all[s, pred]

                # update distances and predecessors
                dists[neigh_stack_ind,
                      s] = cost_and_angle + edge_cost[neigh_stack_ind, s]
                preds[neigh_stack_ind, s] = pred
    return dists, preds


@jit(nopython=True)
def sp_dag_reversed(
    stack, pos2node, shifts, angles_all, dists, edge_cost, algorithm, *args
):
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

        predecessors = algorithm(in_dists, *args)

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
