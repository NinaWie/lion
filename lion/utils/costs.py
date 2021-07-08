import numpy as np
from lion.utils.general import (bresenham_line, compute_angle_cost, angle)
try:
    # import only required for the watershed transform, so imported
    # only of available in order to reduce dependencies
    from skimage.segmentation import watershed
    from skimage import filters
except ModuleNotFoundError:
    pass


def compute_edge_costs(path, instance):
    """
    Compute the raw edge costs, i.e. the average value of the crossed cells
    Arguments:
        path: list of tuples or 2D numpy array with path coordinates
        instance: 2D np array with resistance values
    Returns:
        A list with the same length as the path, containing the edge cost vals
        (last value is zero because less edges then vertices)
    """
    path = np.array(path)
    e_costs = []
    for p in range(len(path) - 1):
        point_list = bresenham_line(*tuple(path[p]), *tuple(path[p + 1]))
        e_costs.append(
            np.mean([instance[i, j] for (i, j) in point_list[1:-1]])
        )
    # to make it the same size as other costs
    e_costs.append(0)
    return np.array(e_costs)


def compute_geometric_costs(path, instance, edge_weight=0):
    """
    Compute geometric costs along the path
    Arguments:
        path: List or array of path corrdinates
        instance: 2D array of resistances
        edge_costs: 1D array, previously computed costs between points along
            the path
    Returns:
        List of geometric edge costs along the path
    """
    path = np.array(path)
    # compute the between point costs (bresenham line between points)
    edge_costs = np.zeros(len(path))
    if edge_weight != 0:
        edge_costs = compute_edge_costs(path, instance) * edge_weight

    geometric_costs = []
    for p in range(len(path) - 1):
        # between point costs
        bresenham_edge_dist = edge_costs[p]
        # compute distance inbetween
        shift_costs = np.linalg.norm(path[p] - path[p + 1])
        # compute geometric edge costs
        geometric_costs.append(
            (
                0.5 *
                (instance[tuple(path[p])] + instance[tuple(path[p + 1])]) +
                shift_costs * bresenham_edge_dist
            )
        )
    geometric_costs.append(0)
    return geometric_costs


def compute_angle_costs(path, angle_norm_factor=np.pi / 2, mode="linear"):
    """
    Compute the normalized angle costs along the path
    Arguments:
        path: list of tuples or 2D numpy array with path coordinates
        instance: 2D np array with resistance values
    Returns:
        List of same length as path containing angle cost values
        (first and last entry 0 because angle at s and t is irrelevant)
    """
    path = np.asarray(path)
    ang_out = [0]
    for p in range(len(path) - 2):
        vec1 = path[p + 1] - path[p]
        vec2 = path[p + 2] - path[p + 1]
        ang_out.append(
            compute_angle_cost(
                angle(vec1, vec2), angle_norm_factor, mode=mode
            )
        )
    ang_out.append(0)

    return ang_out


def compute_raw_angles(path):
    """
    Compute the raw angles along the path (not the cost!)

    Returns:
        List of same length as path containing angle cost values
        (first and last entry 0 because angle at s and t is irrelevant)
    """
    path = np.asarray(path)
    ang_out = [0]
    for p in range(len(path) - 2):
        vec1 = path[p + 1] - path[p]
        vec2 = path[p + 2] - path[p + 1]
        ang_out.append(angle(vec1, vec2))
    ang_out.append(0)
    return ang_out
