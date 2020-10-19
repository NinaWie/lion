import numpy as np
from lion.utils.general import (bresenham_line, compute_angle_cost, angle)
try:
    # import only required for the watershed transform, so imported
    # only of available in order to reduce dependencies
    from skimage.segmentation import watershed
    from skimage import filters
except ModuleNotFoundError:
    pass

__all__ = [
    "inf_downsample", "downsample", "compute_angle_costs",
    "compute_edge_costs", "compute_raw_angles"
]


def inf_downsample(img, factor, func="mean"):
    """
    TODO: merge with downsample method above
    Downsampling function to reduce the size of an inst by a certain factor,
    but replace only non inf values
    """
    x_len_new = img.shape[1] // factor
    y_len_new = img.shape[2] // factor
    new_img = np.zeros(img.shape)
    new_img += np.inf
    pool_func = eval("np." + func)
    for i in range(x_len_new):
        for j in range(y_len_new):
            patch = img[:, i * factor:(i + 1) * factor,
                        j * factor:(j + 1) * factor]
            if np.any(patch < np.inf):
                for k in range(len(new_img)):
                    part = patch[k]
                    new_img[k, i * factor,
                            j * factor] = pool_func(part[part < np.inf])
    return new_img


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
    e_costs = []
    for p in range(len(path) - 1):
        point_list = bresenham_line(
            path[p][0], path[p][1], path[p + 1][0], path[p + 1][1]
        )
        e_costs.append(
            np.mean([instance[i, j] for (i, j) in point_list[1:-1]])
        )
    # to make it the same size as other costs
    e_costs.append(0)
    return e_costs


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
