import numpy as np
from lion.utils.general import (bresenham_line, discrete_angle_costs, angle)
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


def get_seeds(greater_zero, factor):
    """
    Get seeds in grid of every factor pixel
    Arguments:
        greater_zero: binary image indicating where seeds are to be placed
        factor: every factor pixel is a seed
    Returns:
        Array of same shape as greater zero with grid of evenly spaced
        values ranging from 1 to number of seeds
    """
    lab = 0
    x_len, y_len = greater_zero.shape
    seeds = np.zeros(greater_zero.shape)
    omitted = 0
    # consider every factor pixel in each dimension
    for i in np.arange(0, x_len, factor):
        for j in np.arange(0, y_len, factor):
            if greater_zero[i, j]:
                # set seed with new index
                seeds[i, j] = lab
                lab += 1
            else:
                omitted += 1
    return seeds


def watershed_transform(cost_rest, factor, compact=0.01, func="mean"):
    """
    :param mode: all = all combinations in one cluster possible
            --> leading to larger distances
            center = only the center of each cluster can be connected
    """
    pool_func = eval("np." + func)
    # take mean image for clustering TODO: weighted sum?
    img = np.mean(cost_rest, axis=0)

    greater_zero = (img > 0).astype(int)

    # get edge image
    edges = filters.sobel(img)
    # get regular seeds
    seeds = get_seeds(greater_zero, factor)

    w1 = watershed(edges, seeds, compactness=compact)
    # w1 is full watershed --> labels spread over corridor borders
    # but: label 0 also included --> +1 before corridor
    w1_g_zero = (w1 + 1) * greater_zero
    # labels: 0 is forbidden, 1 etc is watershed labels
    labels = np.unique(w1_g_zero)

    new_cost_rest = np.zeros(cost_rest.shape)
    # iterate over labels (except for 0 - forbidden)
    for _, lab in enumerate(labels[1:]):
        x_inds, y_inds = np.where(w1_g_zero == lab)
        for j in range(len(cost_rest)):
            new_cost_rest[j, int(np.mean(x_inds)),
                          int(np.mean(y_inds))] = pool_func(
                              cost_rest[j, x_inds, y_inds]
                          )
    return new_cost_rest


def simple_downsample(img, factor, func="mean"):
    """
    Summarize pixels into on with a certain function
    Arguments:
        img: input 3d Array of costs (first dim: cost classes)
        factor: how many pixels to summarize
        func: pooling function - can be any such as
            np.mean np.min or np.max
    Returns:
        image that is zero everywhere except for the selected points
    """
    x_len_new = img.shape[1] // factor
    y_len_new = img.shape[2] // factor
    new_img = np.zeros(img.shape)
    pool_func = eval("np." + func)
    for i in range(x_len_new):
        for j in range(y_len_new):
            patch = img[:, i * factor:(i + 1) * factor,
                        j * factor:(j + 1) * factor]
            if np.any(patch):
                for k in range(len(new_img)):
                    part = patch[k]
                    if np.any(part):
                        new_img[k, i * factor,
                                j * factor] = pool_func(part[part > 0])
    return new_img


def downsample(img, factor, mode="simple", func="mean", compact=0.01):
    """
    Interface to downsampling possibilities
    """
    if mode == "simple":
        return simple_downsample(img, factor, func=func)
    elif mode == "watershed":
        return watershed_transform(img, factor, compact=compact)
    else:
        raise NotImplementedError


def inf_downsample(img, factor, func="mean"):
    """
    TODO: merge with downsample method above
    Downsampling function to reduce the size of an instance by a certain factor,
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
            discrete_angle_costs(
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
