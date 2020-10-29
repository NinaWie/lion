from numba import jit
import numpy as np
from lion.utils.general import angle, angle_360


def prepare_for_discrete(angles_all, in_edges, out_edges):
    # get unique discrete costs and bounds
    discrete_costs = np.unique(angles_all)
    discrete_costs = discrete_costs[discrete_costs < np.inf]
    d_uni = len(discrete_costs)
    # print("discrete_costs", discrete_costs)
    # get bounds
    # TODO: normalize edges first to use the fast angle function
    angles_raw = np.array(
        [
            [angle(in_edge, out_edge) for in_edge in in_edges]
            for out_edge in out_edges
        ]
    )
    bounds = np.zeros((d_uni, 2))
    prev_max = 0
    for i, dis_cost in enumerate(discrete_costs):
        covered_angles = angles_raw[angles_all == dis_cost]
        in_between = 0.5 * (prev_max + np.min(covered_angles))
        if i > 0:
            bounds[i - 1, 1] = in_between
        prev_max = np.max(covered_angles)
        bounds[i, 0] = in_between  # , round(np.max(covered_angles), 5)]
    bounds[i, 1] = prev_max + 1

    # compute alphas and betas
    ref_edge = [1, 0]
    alphas = np.asarray(
        [round(angle_360(e, ref_edge) + np.pi, 5) for e in in_edges]
    )
    betas = np.array(
        [round(angle_360(e, ref_edge) + np.pi, 5) for e in out_edges]
    )

    tree_index = np.argsort(betas)
    tree_values = betas[tree_index]

    return alphas, tree_index, tree_values, discrete_costs, bounds


@jit(nopython=True)
def plus(a, b):
    """
    Circular plus
    """
    if a + b > 2 * np.pi:
        return a + b - 2 * np.pi
    return a + b


@jit(nopython=True)
def find(key, tree_values, tree_index):
    """
    Will always return the one lower --> recursive binary search
    """
    tree_len = len(tree_values)
    if tree_len == 1:
        return tree_index[0]
    middle = tree_len // 2
    if key < tree_values[middle]:
        return find(key, tree_values[:middle], tree_index[:middle])
    else:
        return find(key, tree_values[middle:], tree_index[middle:])


@jit(nopython=True)
def minus(a, b):
    """
    Circular minus - if smaller zero, then subtract from 2 pi
    """
    if a < b:
        return 2 * np.pi - b + a
    else:
        return a - b
