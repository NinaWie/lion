import numpy as np
import time
import lion.utils.ksp as ut_ksp
from lion.utils.general import get_distance_surface


class KSP:

    def __init__(self, graph):
        self.graph = graph
        try:
            test = self.graph.dists_ba.shape
            test = self.graph.preds_ba.shape
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

    def ksp(self, k, min_dist=20, cost_add=np.inf):
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
        (min_node_dists, min_shift_dists) = self.compute_min_node_dists()

        _, arr_len = min_node_dists.shape
        for _ in range(k - 1):
            # compute the distances of the current paths to all vertices
            # tic_corr = time.time()
            path_points = np.array([p for path in best_paths
                                    for p in path]).astype(int)
            corridor = ut_ksp.fast_dilation(
                path_points, min_node_dists.shape, iters=min_dist
            )

            # add penalty (or inf to exclude regions)
            corridor = cost_add * corridor / np.max(corridor) + 1
            corridor[np.isnan(corridor)] = 1
            feasible_vertices = corridor * min_node_dists

            if ~np.any(feasible_vertices < np.inf):
                return [self.graph.transform_path(p) for p in best_paths]

            # get min vertex
            current_best = np.nanargmin(feasible_vertices.flatten())
            (x2, x3) = current_best // arr_len, current_best % arr_len
            x1 = min_shift_dists[x2, x3]

            # compute path and add to set
            vertices_path = self.graph._combined_paths(
                self.graph.start_inds, self.graph.dest_inds, x1, [x2, x3]
            )
            best_paths.append(vertices_path)

        self.graph.time_logs["ksp"] = round(time.time() - tic, 3)
        if self.graph.verbose:
            print("compute KSP time:", self.graph.time_logs["ksp"])
        return [self.graph.transform_path(p) for p in best_paths]
