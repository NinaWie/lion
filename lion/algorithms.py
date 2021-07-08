"""
Methods to compute the shortest path or multiple shortest paths
given an instance and a configuration file

Main parameters:

instance: 2D numpy array of resistances values (float or int)
          If cells are forbidden, set them to NaN or a value x and specify x
          as cfg["forbidden_val"] = x.

cfg - configuration: Dict with the following neceassay and optional parameters
    + start_inds: list of two cell coordinates
    + dest_inds: list of two cell coordinates

    - forbidden_val: value indicating that a cell is forbidden / outside of
            the project region (can be int, np.nan, np.inf ... )
    - point_dist_min: minimum cell distance of neighboring points (default 3)
    - point_dist_max: minimum cell distance of neighboring points (default 5)
    - angle_weight: how important is the angle (default 0)
    - edge_weight: importantance of costs between points compared to points
            themselves (default 0 --> only the cost at the points matters)
    - max_direction_deviation: maximum deviation from the straight direction
            from start to end. Unit: angle value (in rad). The default is
            pi/2, which is also the maximum possible in the current version.
            In explanation, going sidewards by 90 degrees is possible but
            going backwards is not possible.
    - max_angle: maximum angle at a point (default: pi)
    - angle_cost_function: 'linear' and 'discrete' are implemented
    - memory_limit: Maximum number of edges that is allowed (default: 50 Mio)
            If the number of edges is higher, an iterative procedure is used.
            Relation of edges to actual memory:
            For single SP, space for two float arrays with #edges cells are
                allocated -> 2 * #edges * 8 bytes (float64) = 16 x #edges bytes
            For multi SP there will be four float arrays with #edges cells
                -> 32 x #edges bytes
    - pipeline: List of decreasing positive integers, ending with 1
            The pipeline in an iterative approach defines the downsampling
            factors for each step. By default, it is set automatically based on
            the memory limit. It can however be set manually as well, e.g.
            [3,1] means downsample by factor of 3, compute optimal
            path, reduce region of interest to a corridor around
            optimal path (corridor width is computed automatically based on the
            memory_limit) then downsample by factor of 1 (aka full resolution).
            There is no support for setting the corridor width manually because
            it does not make sense to make it smaller than it could be
    - between_points_allowed: If True, then forbidden areas can still be
            traversed, i.e. two points can be placed around a forbidden area
            If False, then forbidden areas can not be traversed either
    - diversity_threshold:
        FOR KSP.ksp:
            Minimum diversity of next path from previous paths in cell
            distance. E.g. if thresh = 200, each path will be at least 200
            cells away at one point from each other path.
            If None, it is set by default to 1/20 of the instance size

        FOR KSP.min_set_intersection:
            maximum intersection of the new path with all previous points
            Must be between 0 and 1. E.g. if 0.2, then at most 20% of cells
            are shared between one path and the other paths
"""

import numpy as np
from lion.angle_graph import AngleGraph
from lion.ksp import KSP
import lion.utils.costs as ut_cost
import lion.utils.general as ut_general
import time
import logging

logger = logging.getLogger(__name__)


def _initialize_graph(instance, cfg):
    """
    Auxiliary method to preprocess the instance and initialize the graph
    @params:
        instance: 2D numpy array of resistances values (float or int)
                  If cells are forbidden, set them to NaN or a value x and
                  specify x in cfg["forbidden_val"].
        cfg: configuration as specified on top of this file
    @returns:
        graph: AngleGraph object initialized with instance
        cfg: updated configuration file
    """
    forbidden_val = cfg.get("forbidden_val", np.nan)
    logger.info(f"forbidden val: {forbidden_val}")

    # make forbidden region array
    project_region = np.ones(instance.shape)
    project_region[np.isnan(instance)] = 0
    project_region[instance == forbidden_val] = 0

    # instance must not contain zeros
    non_zero_add = 1e-8
    instance += non_zero_add

    # compute maximum values without forbidden weight
    normal_vals = instance[
        np.logical_and(instance != forbidden_val, ~np.isnan(instance))]
    assert np.all(
        normal_vals < np.inf
    ), "check forbidden_val parameter in cfg, it is\
         not inf but there are inf values in array"

    # fill values in instance
    instance[project_region == 0] = np.max(normal_vals)
    # normalize instance by maximum value (excluding forbidden areas)
    # normalization is necessary to balance angle- and resistance-costs
    instance = instance / np.max(normal_vals)
    assert np.min(instance) > 0 and np.isclose(
        np.max(instance), 1
    ), "Minimum must be greater than zero and maximum ~1 after normalizing"

    # initialize graph
    graph = AngleGraph(instance, project_region)

    return graph, cfg


def optimal_route(instance, cfg):
    """
    Compute the (angle-) optimal path through a grid
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        see top of file for more details
    @returns:
        a single optimal path (list of X Y coordinates)
        or empty list if no path exists
    """
    # initialize graph
    graph, cfg = _initialize_graph(instance, cfg)

    # Put in cfg that we use the raster-routing with geometric costs
    cfg["geometric_route"] = True
    # set the ring to the 8-neighborhood
    cfg["point_dist_min"] = 0.9
    cfg["point_dist_max"] = 1.5

    # compute path
    tic_raster = time.time()
    path = graph.single_sp(**cfg)

    logger.info(f"Overall timefor optimal route: {time.time() - tic_raster}")
    logger.info(f"time logs: {graph.time_logs}")

    return path


def optimal_point_spotting(
    instance, cfg, corridor=None, k=1, algorithm=KSP.ksp
):
    """
    Compute the (angle-) optimal point spotting
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        corridor: relevant region --> either
            - a path, e.g. output of optimal_route, or
            - a 2D array with 0: do not consider, 1: consider
    @returns:
        a single optimal path of points (list of X Y coordinates)
        or empty list if no path exists
    """
    # initialize graph
    graph, cfg = _initialize_graph(instance, cfg)
    # initial project region
    original_inst = graph.edge_inst.copy()
    original_corr = (graph.instance < np.inf).astype(int)
    # initialize corridor
    if corridor is None:
        corridor = np.ones(original_corr.shape)

    # estimate how many edges the graph will have:
    graph.set_shift(cfg["start_inds"], cfg["dest_inds"], **cfg)
    orig_shifts = len(graph.shift_tuples)
    instance_vertices = np.sum(original_corr * corridor > 0)

    mem_limit = cfg.get("memory_limit", 5e7)
    # define pipeline
    pipeline = cfg.get(
        "pipeline",
        ut_general.get_pipeline(instance_vertices, orig_shifts, mem_limit)
    )
    assert all(
        pipeline[i] > pipeline[i + 1] for i in range(len(pipeline) - 1)
    ), "pipeline must consist of decreasing downsampling factors"
    assert pipeline[
        -1] == 1, "last factor in pipeline must be 1 (= no downsampling)"

    # execute pipeline
    logger.info(f"Pipeline set to: {pipeline}")

    # execute iterative shortest path computation
    for pipe_step, factor in enumerate(pipeline):
        assert isinstance(factor, int) or float(factor).is_integer(
        ), "downsampling factors in pipeline must be integers"
        logger.info(
            f"---------- Start {pipe_step+1}th step {factor} ---------------"
        )
        # rescale and set parameters accordingly
        corridor = (corridor * original_corr > 0).astype(int)
        current_instance, current_corridor, current_cfg = ut_general.rescale(
            original_inst, corridor, cfg, factor
        )
        # run shortest path computation
        graph = AngleGraph(current_instance, current_corridor)
        if k > 1:
            paths = _run_ksp(graph, current_cfg, k, algorithm=algorithm)
            paths = [np.array(path) * factor for path in paths]
        else:
            path = graph.single_sp(**current_cfg)
            paths = [np.asarray(path) * factor]
        logger.debug(f"got {len(paths)} paths in this step")

        # compute next corridor
        if pipe_step < len(pipeline) - 1 and len(paths[0]) > 0:
            corridor = ut_general.pipeline_corridor(
                paths, instance.shape, orig_shifts, mem_limit,
                pipeline[pipe_step + 1]
            )
    if k == 1:
        return path
    else:
        return paths


# ----------------------------------- KSP  ---------------------------------
def _run_ksp(graph, cfg, k, algorithm=KSP.ksp):
    """
    Build the shortest path trees and compute k diverse shortest paths
    Arguments:
        graph: AngleGraph object that was initialized with the instance
               (as passed from ksp_routes or ksp_points)
        cfg: See beginning of file for possible parameters
        k: Number of diverse alternatives to compute. Attention: Might output
            less than k paths if no further sufficiently diverse path is found
        algorithm: currently implemented in lion/ksp.py :
                KSP.ksp: Interatively find next shortest path that is at
                         least <thresh> cells from the previously found paths
                KSP.min_set_intersection: Iteratively find next shortest path
                         that shares at most <thresh> cells with the previous
    """

    def set_thresh_automatically():
        # helper method to set the diversity threshold dependent on inst size
        if algorithm == KSP.min_set_intersection:
            thresh = 0.3
        else:
            inst_size = min([graph.instance.shape[0], graph.instance.shape[1]])
            thresh = int(inst_size / 10)
        logger.info(f"set diversity treshold automatically to: {thresh}")
        return thresh

    thresh = cfg.get("diversity_threshold", set_thresh_automatically())
    logger.debug(f"diversity threshold is {thresh}")

    # construct sp trees
    tic = time.time()
    _ = graph.sp_trees(**cfg)
    # compute k shortest paths
    ksp_processor = KSP(graph)
    ksp_paths = algorithm(ksp_processor, k, thresh=thresh)
    logger.info(f"Time for run ksp: {time.time() - tic}")
    return ksp_paths


def ksp_routes(instance, cfg, k, algorithm=KSP.ksp):
    """
    Compute the (angle-) optimal k diverse shortest paths through a grid
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        k: number of paths to compute
        algorithm: see doc of run_ksp
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    # initialize graph
    graph, cfg = _initialize_graph(instance, cfg)

    # Put in cfg that we use the raster-routing with geometric costs
    cfg["geometric_route"] = True
    # set the ring to the 8-neighborhood
    cfg["point_dist_min"] = 1
    cfg["point_dist_max"] = 1.5

    # run algorithm
    return _run_ksp(graph, cfg, k, algorithm=algorithm)


def ksp_points(instance, cfg, k, algorithm=KSP.ksp):
    """
    Compute the (angle-) optimal k diverse shortest path of pointS
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        k: number of paths to compute
        algorithm: see doc of run_ksp
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    return optimal_point_spotting(instance, cfg, k=k, algorithm=algorithm)
