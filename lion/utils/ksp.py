import numpy as np
from numba import jit
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True)
def compute_eucl(path1, path2, mode="mean"):
    """
    Compute Euclideean distance between two paths
    @param mode: if "mean", then compute the average of the minimum distances
                of all vertices of path1 to path2 and the other way round,
                if "max" then the maximum of minimum distances
    @returns: float indicating the distance
    """
    min_dists_out = np.zeros(len(path1))
    for p1 in range(len(path1)):
        min_dists = np.zeros(len(path2))
        for p2 in range(len(path2)):
            min_dists[p2] = np.linalg.norm(path1[p1] - path2[p2])
        min_dists_out[p1] = np.min(min_dists)
    if mode == "mean":
        return np.mean(min_dists_out)
    elif mode == "max":
        return np.max(min_dists_out)
    else:
        raise NotImplementedError("only mean or max distance are possible")


@jit(nopython=True)
def fast_dilation(path_points, arr_shape, iters=50):
    """
    Compute distance transform - indicates dist of each cell to a set of paths
    @params:
        path_points : 2D array of shape (n, 2) of cells on previous paths
        arr_shape : Tuple of integers, shape of raster
        iters : Maximum distance until which is should be computed
    @returns:
        2D raster of shape arr_shape containing the INVERTED distances:
        0 means more than iters distance, iters means distance 0
    """
    arr = np.zeros(arr_shape)
    if iters > arr_shape[0] and iters > arr_shape[1]:
        # logger.warn("corridor width was larger than instance size")
        return np.ones(arr_shape)
    for i in range(len(path_points)):
        arr[path_points[i][0], path_points[i][1]] = 1

    start_x, end_x = (np.min(path_points[:, 0]), np.max(path_points[:, 0]))
    start_y, end_y = (np.min(path_points[:, 1]), np.max(path_points[:, 1]))
    # todo: must be in array bounds
    for i in range(iters):
        arr_prev = arr.copy()
        for x in range(start_x - 1, end_x + 2):
            for y in range(start_y - 1, end_y + 2):
                if arr_prev[x, y] > 0 or arr_prev[x - 1, y] > 0 or (
                    arr_prev[x + 1, y] > 0
                ) or arr_prev[x, y - 1] > 0 or arr_prev[x, y + 1] > 0:
                    arr[x, y] += 1
        if start_x > 1:
            start_x -= 1
        if end_x < arr_shape[0] - 3:
            end_x += 1
        if start_y > 1:
            start_y -= 1
        if end_y < arr_shape[1] - 3:
            end_y += 1

    return arr


def intersecting_ratio(path_list, current_path, max_intersection):
    """
    Compute whether the new path is intersecting more than max_intersection
    with the paths in path_list

    path_list: list of arrays of shape (n,2) containing X Y coords of paths
    current_path: 2Darray of shape (m,2)
    max_intersection: float between 0 and 1 indicating intersection threshold
    """
    convert_num = np.max(current_path)
    current_path_converted = (
        current_path[:, 0] * convert_num + current_path[:, 1]
    )

    for prev_path_ind in range(len(path_list)):
        # get array
        prev_path = path_list[prev_path_ind]
        prev_path_converted = prev_path[:, 0] * convert_num + prev_path[:, 1]
        _, inds_intersection, _ = np.intersect1d(
            current_path_converted,
            prev_path_converted,
            assume_unique=True,
            return_indices=True
        )
        # delete the ones that have already been found
        current_path_converted = np.delete(
            current_path_converted, inds_intersection
        )

        # If above threshold already, then return
        if 1 - len(current_path_converted
                   ) / len(current_path) > max_intersection:
            return False

    # For all paths the intersection has stayed sufficiently low
    return True


def path_distance(p1, p2, mode="jaccard"):
    """
    Compute the distance between two paths
    NOTE: all modes in this method are valid metrics
    Arguments:
        p1,p1: two paths (lists of coordinates!)
        mode: jaccard: jaccard distance (1-IoU)
            eucl_mean: from all min eucl distances, take mean
            eucl_max: from all min eucl distances, take max
    """
    if mode == "jaccard":
        s1 = set([tuple(p) for p in p1])
        s2 = set([tuple(p) for p in p2])
        # s1,s2 = (set(list(p1)),set(list(p2)))
        return 1 - len(s1.intersection(s2)) / len(s1.union(s2))
    elif mode.startswith("euc"):
        p1 = np.array(p1).astype("float")
        p2 = np.array(p2).astype("float")
        eucl_mode = mode.split("_")[1]
        return max(
            [
                compute_eucl(p1, p2, mode=eucl_mode),
                compute_eucl(p2, p1, mode=eucl_mode)
            ]
        )
    else:
        raise NotImplementedError(
            "mode " + mode + " wrong, not implemented yet"
        )


def pairwise_dists(collected_coords, mode="jaccard"):
    nr_paths = len(collected_coords)
    dists = np.zeros((nr_paths, nr_paths))
    for i in range(nr_paths):
        for j in range(i, nr_paths):
            dists[i, j] = path_distance(
                collected_coords[i], collected_coords[j], mode=mode
            )
            dists[j, i] = dists[i, j]
    return dists
