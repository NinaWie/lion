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
"""

import numpy as np
from lion.angle_graph import AngleGraph
from lion.ksp import KSP
import lion.utils.costs as ut_cost
import time
import logging
import sys

logger = logging.getLogger(__name__)

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
    logger.info(f"forbidden val: {forbidden_val}")

    # make forbidden region array
    project_region = np.ones(instance.shape)
    project_region[np.isnan(instance)] = 0
    project_region[instance == forbidden_val] = 0

    # normalize instance -- necessary to have comparable angle weight
    normal_vals = instance[
        np.logical_and(instance != forbidden_val, ~np.isnan(instance))]
    instance = (instance - np.min(normal_vals)
                ) / (np.max(normal_vals) - np.min(normal_vals))

    # modify instance to have a 3-dimensional input as required
    instance = np.array([instance])

    # init graph
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

    # set the ring to the 8-neighborhood
    cfg["pylon_dist_min"] = 0.9
    cfg["pylon_dist_max"] = 1.5

    # compute path
    tic_raster = time.time()
    path, _, _ = graph.single_sp(**cfg)

    logger.info(f"Overall timefor optimal route: {time.time() - tic_raster}")
    logger.info(f"time logs: {graph.time_logs}")

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
    # pylon spotting
    path, _, _ = graph.single_sp(**cfg)
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
            inst_size = min(
                [
                    graph.hard_constraints.shape[0],
                    graph.hard_constraints.shape[1]
                ]
            )
            thresh = int(inst_size / 20)
        logger.info(f"set diversity treshold automatically to: {thresh}")

    # construct sp trees
    tic = time.time()
    _ = graph.sp_trees(**cfg)
    # compute k shortest paths
    ksp_processor = KSP(graph)
    ksp_out = algorithm(ksp_processor, k, thresh=thresh)
    # extract path itself
    ksp_paths = [k[0] for k in ksp_out]
    logger.info(f"Time for run ksp: {time.time() - tic}")
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


def _compute_costs(instance, path, edge_weight=0):
    # compute angle costs
    angles = ut_cost.compute_raw_angles(path)[:-1]
    # compute the cable costs (bresenham line between pylons)
    edge_costs = np.zeros(len(path))
    if edge_weight != 0:
        edge_costs = ut_cost.compute_edge_costs(path, instance)

    # compute the geometric path costs TODO: after merge, add next 2 lines
    # path_costs = ut_cost.compute_geometric_costs(
    #     path, instance, edge_costs * edge_weight
    # )
    path = np.asarray(path)
    geometric_costs = []
    for p in range(len(path) - 1):
        # compute distance inbetween
        shift_costs = np.linalg.norm(path[p] - path[p + 1])
        # compute geometric edge costs
        geometric_costs.append(
            shift_costs *
            (0.5 * (instance[tuple(path[p])] + instance[tuple(path[p + 1])]))
        )

    return angles, geometric_costs
