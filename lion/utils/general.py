import numpy as np
from numba import jit
import logging
from scipy.ndimage.morphology import distance_transform_edt

logger = logging.getLogger(__name__)


def normalize(instance):
    """
    0-1 normalization of values of instance
    """
    return (instance -
            np.min(instance)) / (np.max(instance) - np.min(instance))


@jit(nopython=True)
def rescale_instance(img, scale_factor):
    """
    Scale down image by a factor
    Arguments:
        img: numpy array of any dimension
        scale_factor: integer >= 1
    Returns:
        numpy array with 1/scale_factor size along each dimension
    """
    if scale_factor == 1:
        return img
    x_len_new = img.shape[0] // scale_factor
    y_len_new = img.shape[1] // scale_factor
    new_img = np.zeros((x_len_new, y_len_new))
    for i in range(x_len_new):
        for j in range(y_len_new):
            patch = img[i * scale_factor:(i + 1) * scale_factor,
                        j * scale_factor:(j + 1) * scale_factor]
            new_img[i, j] = np.mean(patch)
    return new_img


def rescale(instance, corridor, cfg, factor):
    """
    Prepare instance and config for the next pipeline step

    Arguments:
        instance: 2D array with resistances
        corridor: 2D array with binary (0=forbidden, 1=feasible)
        cfg: configuration with start, dest and point distances
        factor: Int > 0 : factor by which to downsample
    """
    current_cfg = cfg.copy()
    if factor > 1:
        # rescale instances
        current_instance = rescale_instance(instance, factor)
        current_corridor = (
            rescale_instance(corridor.astype(float), factor) > 0
        ).astype(int)
        # downscale start and dest
        try:
            current_cfg["start_inds"] = (np.array(cfg["start_inds"]) /
                                         factor).astype(int)
            current_cfg["dest_inds"] = (np.array(cfg["dest_inds"]) /
                                        factor).astype(int)
            # make sure start and dest are in project region
            current_corridor[tuple(current_cfg["start_inds"])] = 1
            current_corridor[tuple(current_cfg["dest_inds"])] = 1
        except KeyError:
            raise RuntimeError(
                "configuration must entail start and destination coordinates"
            )
            # downscale the KSP threshold if necessary
        if "diversity_threshold" in current_cfg.keys(
        ) and current_cfg["diversity_threshold"] > 1:
            current_cfg["diversity_threshold"
                        ] = cfg["diversity_threshold"] / factor
        # downscale the point distances
        if "point_dist_min" in current_cfg.keys():
            current_cfg["point_dist_min"] = cfg["point_dist_min"] / factor
            current_cfg["point_dist_max"] = cfg["point_dist_max"] / factor
        return current_instance, current_corridor, current_cfg
    else:
        return instance, corridor, cfg


def get_donut(radius_low, radius_high):
    """
    Compute all indices of points in donut around (0,0)
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :returns: tuples of indices of points with radius between radius_low
    and radius_high around (0, 0)
    """
    img_size = int(radius_high + 10)
    # xx and yy are 200x200 tables containing the x and y coordinates as values
    # mgrid is a mesh creation helper
    xx, yy = np.mgrid[-img_size:img_size, -img_size:img_size]
    # circle equation
    circle = (xx)**2 + (yy)**2
    # donuts contains 1's and 0's organized in a donut shape
    # you apply 2 thresholds on circle to define the shape
    donut = np.logical_and(
        circle <= (radius_high**2), circle >= (radius_low**2)
    )
    pos_x, pos_y = np.where(donut > 0)
    return pos_x - img_size, pos_y - img_size


def angle(vec1, vec2, normalize=True):
    """
    Compute angle between two vectors
    :params vec1, vec2: two 1-dim vectors of same size, can be lists or array
    :returns angle
    """
    # make array and normalize
    if normalize:
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
    # compute angle
    angle = np.arccos(np.clip(np.dot(vec1, vec2), -1, 1))
    # want to use full 360 degrees
    if np.sin(angle) < 0:
        angle = 2 * np.pi - angle
    # can still be nan if v1 or v2 is 0
    if np.isnan(angle):
        return 0
        # raise ValueError("angle is nan, check whether vec1 or vec2 = 0")
    return angle


def get_donut_vals(donut_tuples, vec):
    """
    compute the angle between edges defined by donut tuples
    :param donut_tuples: list of pairs of tuples, each pair defines an edge
    going from [(x1,y1), (x2,y2)]
    :param vec: vector to compute angle with
    :returns: list of same length as donut_tuples with all angles
    """
    return [angle(tup, vec) + 0.1 for tup in donut_tuples]


def get_half_donut(radius_low, radius_high, vec, max_deviation=0.5 * np.pi):
    """
    Returns only the points with x >= 0 of the donut points (see above)
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :returns: tuples of indices of points with radius between radius_low
    and radius_high around (0, 0)
    """
    pos_x, pos_y = get_donut(radius_low, radius_high)
    new_tuples = []
    for i, j in zip(pos_x, pos_y):
        # compute angle
        ang = angle([i, j], vec)
        # add all valid ones
        if ang <= max_deviation:
            new_tuples.append((i, j))
    return new_tuples


def compute_angle_cost(ang, max_angle, mode="linear"):
    """
    Implementation of different angle cost functions:
        linear: cost increases linearly with the angle
        discrete: discrete cost classes, e.g. 0.3 for all small angles, then 1
        quadratic could be another option, just definition of a function
    Arguments:
        ang: float between 0 and pi, angle between edges
        max_angle: maximum angle cutoff
        mode: "norm" or "discrete"
    returns: angle costs
    Here computed as Stefano said: up to 30 degrees + 50%, up to 60 degrees
    3 times the cost, up to 90 5 times the cost --> norm: 1.5 / 5 = 0.3
    """
    # TODO: 3 times technical costs for example
    if mode == "linear":
        return ang / max_angle
    elif mode == "discrete":
        if ang >= max_angle:
            return np.inf
        elif ang <= 0.1:
            return 0
        elif ang <= np.pi / 6:
            return 0.3
        elif ang <= np.pi / 3:
            return 0.6
        else:
            return 1
    elif mode == "squared":
        return (ang / max_angle)**2
    else:
        raise NotImplementedError


def angle_360(vec1, vec2, normalize=True):
    if normalize:
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
        # normalize
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
    # dot product and determinant
    x1, y1 = vec1
    x2, y2 = vec2
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)
    return angle


def get_lg_donut(
    radius_low,
    radius_high,
    vec,
    max_direction_deviation,
    max_angle=np.pi / 4
):
    """
    Compute all possible combinations (tuples) of edges in restricted angle
    :param radius_low: minimum radius
    :param radius_high: maximum radius
    :param vec: direction vector
    :returns: list with entries [[edge1, edge2, cost of angle between them]]
    where costs are normalized values between 0 and 1
    """
    donut = get_donut(radius_low, radius_high)
    tuple_zip = list(zip(donut[0], donut[1]))
    linegraph_tuples = []
    for (i, j) in tuple_zip:
        # if in incoming half
        if i * vec[0] + j * vec[1] <= 0:
            for (k, l) in tuple_zip:
                ang = angle([-k, -l], [i, j])
                # if smaller max angle and general outgoing half
                if ang <= max_angle and k * vec[0] + l * vec[1] >= 0:
                    angle_norm = compute_angle_cost(ang, max_angle)
                    linegraph_tuples.append([[i, j], [k, l], angle_norm])
    return linegraph_tuples


def get_path_lines(cost_shape, paths):
    """
    Given a list of paths, compute continous lines in an array of cost_shape
    :param cost_shape: desired 2-dim output shape of array
    :param paths: list of paths of possibly different lengths, each path is
    a list of tuples
    :returns: 2-dim binary of shape cost_shape where paths are set to 1
    """
    path_dilation = np.zeros(cost_shape)
    for path in paths:
        # iterate over path nodes
        for i in range(len(path) - 1):
            line = bresenham_line(*path[i], *path[i + 1])
            # set all pixels on line to 1
            for (j, k) in line:
                path_dilation[j, k] = 1
    return path_dilation


def get_pipeline(num_vertices, num_shifts, mem_limit):
    """
    Compute the optimal downsampling factors in an iterative pipeline approach
    Arguments:
        num_vertices: Int, number of vertices in the graph
        num_shifts: Int, number of neighbors for each vertex
        mem_limit: Int, maximal number of edges allowed
    """
    factor = 1
    # return first one that fits in mem_limit
    while (num_vertices * num_shifts) / factor**4 > mem_limit:
        factor += 1
    return np.arange(factor, 0, -1)


def pipeline_corridor(paths, out_shape, n_shifts, mem_limit, next_factor):
    """
    Get the next corridor in a pipeline (automatically calibrate the corridor
    width based on the next sampling factor)

    Arguments:
        path_points: list of arrays of shape (n, 2) containing the points on one
                or more paths that have been found in the previous iteration
        out_shape: Tuple, shape of array that will be the output corridor
        n_shifts: Int, Number of neighbors per vertex (based on point_dist_min
                and point_dist_max)
        mem_limit: Int, Maximum number of edges
        next_factor: Int, downsampling factor in the upcoming next iteration
    """
    # close gap between points on path with bresenham line
    distance_transform = np.ones(out_shape)
    for path_points in paths:
        for i in range(len(path_points) - 1):
            line = bresenham_line(*path_points[i], *path_points[i + 1])
            for (x, y) in line:
                distance_transform[x, y] = 0
    # compute distance transform
    # distance_transform[x,y] = min distance of cell (x,y) to the path(s)
    distance_transform = distance_transform_edt(distance_transform)
    # check how many pixels you get when setting the corridor to all cells with
    # distance less than 20 (in order to test how many cells you get then)
    test_corridor_width = 20
    test_corridor = (distance_transform < test_corridor_width).astype(int)
    # Estimate the number of edges that one will get with the 20-cell corridor:
    # #edges = #pixels_in_corridor x #neighbors / factor**4 because a lower
    # resolution reduces #pixels quadratically and #neighbors quadtratically
    estimated_edges_20 = (np.sum(test_corridor) * n_shifts) / (next_factor**4)
    # (mem_limit / estimated_edges_20) is then the factor how much more or less
    # cells we can include in the corridor. E.g. With 20 we get 100 edges, but
    # mem_limit = 1000 --> we can use 200 instead of 20 as the corridor width
    # (but must be at least 10)
    optimal_corridor_width = max(
        [test_corridor_width * mem_limit / estimated_edges_20, 3]
    )
    # get the corridor with all cells closer than optimal_corridor_width
    corridor = (distance_transform <= optimal_corridor_width).astype(int)
    logger.info(
        f"Next corridor around path was set to width {optimal_corridor_width}"
    )
    return corridor


def bresenham_line(x0, y0, x1, y1):
    """
    Finds the cell indices on a straight line between two raster cells
    """
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        if steep:
            line.append([y, x])
        else:
            line.append([x, y])

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line
