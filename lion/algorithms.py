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
    - pylon_dist_min: minimum cell distance of neighboring pylons (default 3)
    - pylon_dist_max: minimum cell distance of neighboring pylons (default 5)
    - angle_weight: how important is the angle (default 0)
    - edge_weight: importantance of cable costs compared to pylons (default 0)
    - max_angle: maximum deviation in angle from the straight connection from
            start to end (default: pi/2)
    - max_angle_lg: maximum angle at a pylon (default: pi)
    - angle_cost_function: 'linear' and 'discrete' are implemented
    - memory_limit: default is 1 trillion, if the number of edges is higher,
            an iterative pipeline procedure is used
    - pipeline: pipeline in iterative approach is set automatically based on
            the memory limit. It can however be set manually as well, e.g.
            [(4,50), (2,10)] means downsample by factor of 4, compute optimal
            path, reduce region of interest to a corridor of width 50 around
            optimal path, again downsample by factor of 2
    - cable_allowed: If True, then forbidden areas can still be traversed with
            a cable (only placing a pylon is forbidden)
            If False, then forbidden areas can not be traversed either
"""

import numpy as np
from lion.angle_graph import AngleGraph
from lion.ksp import KSP
import lion.utils.general as ut_general
import time

VERBOSE = 0

__all__ = [
    "optimal_route", "optimal_pylon_spotting", "ksp_routes", "ksp_pylons"
]


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
    if VERBOSE:
        print("forbidden val", forbidden_val)

    # make forbidden region array
    project_region = np.ones(instance.shape)
    project_region[np.isnan(instance)] = 0
    project_region[instance == forbidden_val] = 0

    # normalize instance -- necessary to have comparable angle weight
    normal_vals = instance[
        np.logical_and(instance != forbidden_val, ~np.isnan(instance))]
    assert np.all(
        normal_vals < np.inf
    ), "check forbidden_val parameter in cfg, it is\
         not inf but there are inf values in array"

    # fill values in instance
    instance[project_region == 0] = np.max(normal_vals)
    instance = (instance - np.min(normal_vals)
                ) / (np.max(normal_vals) - np.min(normal_vals))

    # init graph
    graph = AngleGraph(instance, project_region, verbose=VERBOSE)

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

    # set the ring to the 8-neighborhood
    cfg["pylon_dist_min"] = 0.9
    cfg["pylon_dist_max"] = 1.5

    # compute path
    tic_raster = time.time()
    path, _, _ = graph.single_sp(**cfg)

    if VERBOSE:
        print(
            "Overall timefor optimal route",
            time.time() - tic_raster, graph.time_logs
        )

    return path


def optimal_pylon_spotting(instance, cfg, corridor=None):
    """
    Compute the (angle-) optimal pylon spotting
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        corridor: relevant region --> either
            - a path, e.g. output of optimal_route, or
            - a 2D array with 0: do not consider, 1: consider
    @returns:
        a single optimal path of pylons (list of X Y coordinates)
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

    # execute pipeline
    if VERBOSE:
        print("chosen pipeline:", pipeline)

    # execute iterative shortest path computation
    for pipe_step, factor in enumerate(pipeline):
        # rescale and set parameters accordingly
        corridor = (corridor * original_corr > 0).astype(int)
        current_instance, current_corridor, current_cfg = ut_general.rescale(
            original_inst, corridor, cfg, factor
        )
        # run shortest path computation
        graph = AngleGraph(current_instance, current_corridor, verbose=VERBOSE)
        path, _, _ = graph.single_sp(**current_cfg)

        # compute next corridor
        if pipe_step < len(pipeline) - 1 and len(path) > 0:
            path = np.array(path) * factor
            corridor = ut_general.pipeline_corridor(
                path, instance.shape, orig_shifts, mem_limit,
                pipeline[pipe_step + 1]
            )
    return path


# ----------------------------------- KSP  ---------------------------------
def _run_ksp(graph, cfg, k, algorithm=KSP.ksp, thresh=None):
    """
    Build the shortest path trees and compute k diverse shortest paths
    Arguments:
        graph: AngleGraph object that was initialized with the instance
               (as passed from ksp_routes or ksp_pylons)
        cfg: See beginning of file for possible parameters
        k: Number of diverse alternatives to compute. Attention: Might output
            less than k paths if no further sufficiently diverse path is found
        algorithm: currently implemented in lion/ksp.py :
                KSP.ksp: Interatively find next shortest path that is at
                         least <thresh> cells from the previously found paths
                KSP.min_set_intersection: Iteratively find next shortest path
                         that shares at most <thresh> cells with the previous
        thresh: FOR KSP.ksp:
                Minimum diversity of next path from previous paths in cell
                distance. E.g. if thresh = 200, each path will be at least 200
                cells away at one point from each other path.
                If None, it is set by default to 1/20 of the instance size

                FOR KSP.min_set_intersection:
                maximum intersection of the new path with all previous points
                Must be between 0 and 1. E.g. if 0.2, then at most 20% of cells
                are shared between one path and the other paths
    """
    if thresh is None:
        if algorithm == KSP.min_set_intersection:
            thresh = 0.3
        else:
            # set appropriate threshold automatically
            inst_size = min([graph.instance.shape[0], graph.instance.shape[1]])
            thresh = int(inst_size / 20)
        if VERBOSE:
            print("set diversity treshold automatically to", thresh)

    # construct sp trees
    tic = time.time()
    _ = graph.sp_trees(**cfg)
    # compute k shortest paths
    ksp_processor = KSP(graph)
    ksp_out = algorithm(ksp_processor, k, thresh=thresh)
    # extract path itself
    ksp_paths = [k[0] for k in ksp_out]
    if VERBOSE:
        print("Overall timefor run ksp", time.time() - tic)
    return ksp_paths


def ksp_routes(instance, cfg, k, thresh=None, algorithm=KSP.ksp):
    """
    Compute the (angle-) optimal k diverse shortest paths through a grid
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        k: number of paths to compute
        thresh: see doc of run_ksp
        algorithm: see doc of run_ksp
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    # initialize graph
    graph, cfg = _initialize_graph(instance, cfg)

    # set the ring to the 8-neighborhood
    cfg["pylon_dist_min"] = 1
    cfg["pylon_dist_max"] = 1.5

    # run algorithm
    return _run_ksp(graph, cfg, k, thresh=thresh, algorithm=algorithm)


def ksp_pylons(instance, cfg, k, thresh=None, algorithm=KSP.ksp):
    """
    Compute the (angle-) optimal k diverse shortest path of PYLONS
    @params:
        instance: 2D np array of resistances (see details top of file)
        cfg: config dict, must contain start and dest (see details top of file)
        k: number of paths to compute
        thresh: see doc of run_ksp
        algorithm: see doc of run_ksp
    @returns:
        A list of paths (each path is again a list of X Y coordinates)
    """
    # initialize graph
    graph, cfg = _initialize_graph(instance, cfg)

    # run algorithm
    return _run_ksp(graph, cfg, k, thresh=thresh, algorithm=algorithm)
