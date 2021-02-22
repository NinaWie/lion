import numpy as np
import time
from scipy.ndimage.morphology import distance_transform_edt
import lion.utils.ksp as ut_ksp
import logging

logger = logging.getLogger(__name__)


class KSP:

    def __init__(self, graph):
        self.graph = graph
        try:
            test = self.graph.dists_ba.shape  # noqa
            test = self.graph.preds_ba.shape  # noqa
        except AttributeError:
            raise RuntimeError(
                "Cannot initialize KSP object with a graph without"
                "shortest path trees in both directions!"
            )

    def compute_min_node_dists(self):
        """
        Eppstein's algorithm: Sum up the two SP treest and iterate
        """
        # sum both dists_ab and dists_ba, inst and edges are counted twice!
        double_counted = self.graph.edge_cost.copy()
        # must set to zero in order to avoid inf - inf
        double_counted[double_counted == np.inf] = 0
        summed_dists = (
            self.graph.dists + self.graph.dists_ba - double_counted
        )
        # mins along outgoing edges
        min_shift_dists = np.argmin(summed_dists, axis=1)

        # project back to 2D:
        min_dists_2d = np.zeros(self.graph.instance.shape) + np.inf
        min_shifts_2d = np.zeros(self.graph.instance.shape)
        x_inds = self.graph.stack_array[:, 0]
        y_inds = self.graph.stack_array[:, 1]
        min_shifts_2d[x_inds,
                      y_inds] = min_shift_dists[self.graph.pos2node[x_inds,
                                                                    y_inds]]
        min_dists_2d[x_inds,
                     y_inds] = summed_dists[self.graph.pos2node[x_inds,
                                                                y_inds],
                                            min_shift_dists]
        return min_dists_2d, min_shifts_2d

    def ksp(self, k, thresh=20, cost_add=np.inf):
        """
        Fast KSP method as tradeoff between diversity and cost
        (add additional cost to the paths found so far)
        Arguments:
            self.graph.start_inds, self.graph.dest_inds, k: see other methods
            min_dist: minimum distance from the previously computed paths
                (or distance in which we add a penalty)
            cost_add: cost_add of 0.05 means that 5% of the best path costs is
                the maximum costs that are added
        Returns:
            List of ksp with costs
        """
        tic = time.time()
        best_paths = [self.graph.best_path]

        # get Eppstein distances
        (min_node_dists, min_shift_dists) = self.compute_min_node_dists()

        # make auxiliary array for distance transform
        aux_arr = np.ones(min_node_dists.shape)
        for (x, y) in best_paths[0]:
            aux_arr[x, y] = 0

        _, arr_len = min_node_dists.shape
        for _ in range(k - 1):
            # use distance transform to compute the distances to the paths
            distance_transform = distance_transform_edt(aux_arr)
            corridor = thresh - distance_transform
            corridor[corridor < 0] = 0

            # add penalty (or inf to exclude regions)
            corridor[corridor > 0
                     ] = cost_add * corridor[corridor > 0] / np.max(corridor)
            feasible_vertices = (corridor + 1) * min_node_dists

            if ~np.any(feasible_vertices < np.inf):
                return best_paths

            # get min vertex
            current_best = np.nanargmin(feasible_vertices.flatten())
            (x2, x3) = current_best // arr_len, current_best % arr_len
            x1 = min_shift_dists[x2, x3]

            # compute path and add to set
            vertices_path = self.graph._combined_paths(x1, [x2, x3])
            best_paths.append(vertices_path)
            for (x, y) in vertices_path:
                aux_arr[x, y] = 0

        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        logger.debug(f"compute KSP time: {self.graph.time_logs['ksp']}")
        return best_paths

    def min_set_intersection(self, k, thresh=0.5):
        """
        Greedy Find KSP algorithm

        Arguments:
            self.graph.start_inds, self.graph.dest_inds: vertices -->
            list with two entries
            k: int: number of paths to output
            max_intersection: ratio of vertices that are allowed to be
            contained in the previously computed SPs
        """
        assert 0 <= thresh <= 1, "threshold for min_set_intersection\
             must be between 0 and 1"

        tic = time.time()

        best_paths = [np.array(self.graph.best_path)]

        # sum both dists_ab and dists_ba, subtract inst because counted twice
        (min_node_dists, min_shift_dists) = self.compute_min_node_dists()

        # argsort
        stack_sorted = np.dstack(
            np.unravel_index(
                np.argsort(min_node_dists.ravel()), min_node_dists.shape
            )
        )[0]
        # iterate over edges from least to most costly
        sorted_dist_prev = 0
        for x, y in stack_sorted:
            sorted_dist = min_node_dists[x, y]
            if sorted_dist == sorted_dist_prev:
                # on the same path as the vertex before
                continue
            if sorted_dist == np.inf:
                break
            sorted_dist_prev = sorted_dist
            s = min_shift_dists[x, y]

            # get shortest path through this node
            # if self.graph.dists_ba[x1, x2, x3] == 0:
            # = 0 for inc edges of self.graph.dest_inds_inds (init of dists_ba)
            # continue
            vertices_path = self.graph._combined_paths(s, [x, y])
            vertices_path = np.array(vertices_path)
            # compute intersection with previous paths
            intersection_low_enough = ut_ksp.intersecting_ratio(
                best_paths, vertices_path, thresh
            )
            # if similarity < threshold, add
            if intersection_low_enough:
                best_paths.append(vertices_path)
                if len(best_paths) >= k:
                    break
        return best_paths
