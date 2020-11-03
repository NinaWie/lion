import lion.utils.general as ut
import lion.utils.costs as ut_cost
import lion.utils.ksp as ut_ksp
from lion.utils.shortest_path import get_algorithm
from lion.fast_shortest_path import (
    sp_dag, sp_dag_reversed, topological_sort_jit, edge_costs
)
import warnings
import numpy as np
import time
import pickle
from numba.typed import List


class AngleGraph():

    def __init__(
        self,
        instance,
        instance_corr,
        edge_instance=None,
        directed=True,
        verbose=1
    ):
        # initialiye edge instance
        self.instance = instance.copy()
        assert np.all(
            self.instance < np.inf
        ), "No infs allowed in instance input to AngleGraph"
        # construct instance with infs where hard constraints are:
        self.instance[np.where(instance_corr == 0)] = np.inf

        if edge_instance is None:
            self.edge_inst = instance
        else:
            self.edge_inst = edge_instance
        self.x_len, self.y_len = instance_corr.shape
        self.time_logs = {}
        self.verbose = verbose
        self.directed = directed

    def set_shift(
        self,
        start,
        dest,
        pylon_dist_min=3,
        pylon_dist_max=5,
        max_angle=np.pi / 2,
        **kwargs
    ):
        """
        Initialize shift variable by getting the donut values

        Arguments:
            start, dest: list containing X and Y coordinate of source / dest
            pylon_dist_min, pylon_dist_max: min and max distance of pylons
            max_angle: Maximum angle of edges to vec
        """
        self.start_inds = np.asarray(start)
        self.dest_inds = np.asarray(dest)
        vec = self.dest_inds - self.start_inds
        shifts = ut.get_half_donut(
            pylon_dist_min, pylon_dist_max, vec, angle_max=max_angle
        )
        shift_angles = [ut.angle_360(s, vec) for s in shifts]
        # sort the shifts
        self.shifts = np.asarray(shifts)[np.argsort(shift_angles)]
        self.shift_tuples = self.shifts

        # determine whether the graph is directed acyclic
        self.is_dag = max_angle <= np.pi / 2

        # construct bresenham lines
        shift_lines = List()
        for shift in self.shifts:
            line = ut.bresenham_line(0, 0, shift[0], shift[1])
            shift_lines.append(np.array(line))
        self.shift_lines = shift_lines

        # construct shift values for diagonals
        self.shift_costs = np.array([np.linalg.norm(s) for s in self.shifts])

    def add_nodes(self):
        """
        Initialize distances and predecessors, sort vertices topologically
        """
        tic = time.time()
        # SORT --> Make stack
        visit_points = (self.instance < np.inf).astype(int)
        initial_pos2node = np.zeros(visit_points.shape) - 1
        # run topologial sort --> fill pos2node
        initial_pos2node, _ = topological_sort_jit(
            self.dest_inds[0], self.dest_inds[1],
            np.asarray(self.shifts) * (-1), visit_points, initial_pos2node, 0
        )
        # build stack from pos2node (sort it)
        self.stack_array = np.dstack(
            np.unravel_index(
                np.argsort(initial_pos2node.ravel()), visit_points.shape
            )
        )[0]
        start_point = np.where(
            np.all(self.stack_array == self.start_inds, axis=1)
        )[0][0]
        self.stack_array = (self.stack_array[start_point:]).astype(int)
        if self.verbose:
            print("time stack construction sort:", round(time.time() - tic, 3))
            print(
                "stack", len(self.stack_array), self.stack_array[0],
                self.stack_array[-1]
            )

        # build pos2node
        self.pos2node = (
            initial_pos2node - start_point +
            len(initial_pos2node[initial_pos2node == -1])
        ).astype(int)
        self.pos2node[self.pos2node < 0] = -1

        # initializes dists and predecessors
        tic = time.time()
        self.dists = np.zeros(
            (len(self.stack_array), len(self.shifts))
        ) + np.inf
        self.dists[0, :] = 0
        self.preds = np.zeros(self.dists.shape) - 1
        self.time_logs["add_nodes"] = round(time.time() - tic, 3)
        self.n_pixels = self.x_len * self.y_len
        self.n_nodes = len(self.stack_array)
        self.n_edges = len(self.shifts) * len(self.dists)
        if self.verbose:
            print("memory taken (dists shape):", self.n_edges)

    def set_edge_costs(
        self,
        layer_classes=["resistance"],
        class_weights=[1],
        angle_weight=0,
        max_angle_lg=np.pi,
        angle_cost_function='linear',
        cable_allowed=True,
        **kwargs
    ):
        """
        Define the resistances / costs to represent on graph edges
        (Combines class-wise resistances and angle weights)

        Arguments:
            layer_classes: List of strings, names of cost categories
            class_weights: List of same length as layer_classes, corresponding
                weights (normalized automatically, can be any positive number)
            angle_weight: Importance of angle costs compared to resistances
                (=0 means only resistance is optimized, =1 means only angles
                are minimized, i.e. output will be straightest line possible)
            max_angle_lg: maximum angle between a adjacent edges on the path
            angle_cost_function: Currently implemented "linear" and "discrete"
                        Function defines the cost per angle, implemented in
                        utils/general.py (function compute_angle_cost)
        """
        tic = time.time()
        assert len(layer_classes) == len(
            class_weights
        ), f"classes ({len(layer_classes)}) and\
            weights({len(class_weights)}) must be of same length!"

        assert 0 <= angle_weight <= 1, "angle weight must be between 0 and 1"
        # set classes
        self.cost_classes = ["angle"] + list(layer_classes)
        # set weights and add angle weight
        self.cost_weights = np.array(
            [angle_weight] +
            list(np.asarray(class_weights) * (1 - angle_weight))
        )
        self.cost_weights = self.cost_weights / np.sum(self.cost_weights)
        if self.verbose:
            print("cost weights", self.cost_weights)

        # set angle weight and already multiply with angles
        self.angle_weight = self.cost_weights[0]
        # in precomute angles, it is multiplied with angle weights
        self.angle_cost_array = self._precompute_angles(
            max_angle_lg, angle_cost_function
        )

        # If it is not allowed to traverse forbidden areas with a cable,
        # transform edge instance accordingly
        if not cable_allowed:
            self.edge_inst[self.instance == np.inf] = np.inf

        self.time_logs["add_all_edges"] = round(time.time() - tic, 3)
        if self.verbose:
            print("instance shape", self.instance.shape)

    def _precompute_angles(self, max_angle_lg, angle_cost_function):
        """
        Helper function to precompute the angle costs for all tuples of edges
        Arguments:
            max_angle_lg: maximum feasible angle
            angle_cost_function: funct to compute cost dependent on the angle
                        currently implemented: linear and one discrete option
        """
        self.angle_cost_function = angle_cost_function
        tic = time.time()

        # normalize (otherwise need to normalize in quadratic loop)
        shift_arrs = [np.asarray(s) for s in self.shifts]
        norm_shifts = [s / np.linalg.norm(s) for s in shift_arrs]
        # compute raw angle values
        angles_raw = np.array(
            [
                [ut.angle(s2, s1, normalize=False) for s1 in norm_shifts]
                for s2 in norm_shifts
            ]
        )
        # compute feasible maximum value
        max_angle = np.max(angles_raw[angles_raw <= max_angle_lg])
        self.angle_norm_factor = max_angle

        # compute angle costs (and normalize)
        slen = len(self.shifts)
        angles_all = np.array(
            [
                [
                    ut.compute_angle_cost(
                        angles_raw[i, j],
                        self.angle_norm_factor,
                        mode=self.angle_cost_function
                    ) for i in range(slen)
                ] for j in range(slen)
            ]
        )
        # greater than 1 (normalized) means infeasible angle
        angles_all[angles_all > 1] = np.inf

        self.time_logs["compute_angles"] = round(time.time() - tic, 3)
        # multiply with angle weights, need to prevent that not inf * 0
        angles_all[angles_all < np.inf
                   ] = angles_all[angles_all < np.inf] * self.angle_weight
        return angles_all

    # --------------------------------------------------------------------
    # SHORTEST PATH COMPUTATION

    def build_source_sp_tree(self, edge_weight=0.2, **kwargs):
        self.edge_weight = edge_weight
        shift_norms = np.array([np.linalg.norm(s) for s in self.shifts])
        if np.any(shift_norms == 1):
            # warnings.warn("Raster approach, EDGE WEIGHT IS SET TO ZERO")
            self.edge_weight = 0

        shift_norms = [np.linalg.norm(s) for s in self.shifts]
        tic = time.time()
        # precompute edge costs
        self.edge_cost = np.zeros(self.preds.shape) + np.inf
        self.edge_cost = edge_costs(
            self.stack_array, self.pos2node, np.array(self.shifts),
            self.edge_cost, self.instance, self.edge_inst, self.shift_lines,
            self.shift_costs, self.edge_weight
        )
        if self.verbose:
            print("Computed edge instance", time.time() - tic)
        tic = time.time()
        # prepare for discrete if it is a discrete angle cost function:
        self.algorithm, self.args = get_algorithm(
            self.angle_cost_function, self.angle_cost_array, self.shifts
        )

        # RUN - either directed acyclic or BF algorithm
        if self.is_dag:
            self.dists, self.preds = sp_dag(
                self.stack_array, self.pos2node, np.array(self.shifts),
                self.angle_cost_array, self.dists, self.preds, self.edge_cost,
                self.algorithm, self.args
            )
        else:
            raise NotImplementedError(
                "Angle shortest path not implemented for cyclic graph.\
                    Please set max_angle <= np.pi / 2 in config"
            )

        self.time_logs["shortest_path"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time edges:", round(time.time() - tic, 3))

    # ----------------------------------------------------------------------
    # REVERSED TREE FOR KSP

    def build_dest_sp_tree(self, source, target):
        """
        Compute costs from dest to all edges
        """
        tic = time.time()

        # initialize dists array
        self.dists_ba = np.zeros(self.dists.shape) + np.inf
        # this time need to set all incoming edges of dest to zero
        # d0, d1 = self.dest_inds
        # for s, (i, j) in enumerate(self.shifts):
        #     pos_index = self.pos2node[d0 + i, d1 + j]
        #     self.dists_ba[pos_index, s] = 0

        # compute distances: new method because out edges instead of in
        self.dists_ba, self.preds_ba = sp_dag_reversed(
            self.stack_array, self.pos2node,
            np.array(self.shifts) * (-1), self.angle_cost_array, self.dists_ba,
            self.edge_cost, self.algorithm, self.args
        )
        self.time_logs["shortest_path_tree"] = round(time.time() - tic, 3)
        if self.verbose:
            print("time shortest_path_tree:", round(time.time() - tic, 3))
        # from lion.utils.plotting import angle_graph_display_dists
        # self.angle_graph_display_dists(self.dists_ba)
        # distance in ba: take IN edges to source, by computing in neighbors
        # take their first dim value (out edge to source) + source val
        (s0, s1) = self.start_inds
        neigh_inds = [self.pos2node[s0 + i, s1 + j] for (i, j) in self.shifts]
        start_dests = [
            self.dists_ba[neigh_inds[s], s] if neigh_inds[s] >= 0 else np.inf
            for s, (i, j) in enumerate(self.shifts)
        ]
        d_ba_arg = np.argmin(start_dests)
        d_ba = np.min(start_dests)

        d_ab = np.min(self.dists[self.pos2node[tuple(self.dest_inds)], :])
        assert np.isclose(
            d_ba, d_ab
        ), "start to dest != dest to start " + str(d_ab) + " " + str(d_ba)

        # compute best path
        self.best_path = np.array(
            ut_ksp.get_sp_dest_shift(
                self.dists_ba,
                self.preds_ba,
                self.pos2node,
                self.dest_inds,
                self.start_inds,
                np.array(self.shifts) * (-1),
                d_ba_arg,
                dest_edge=True
            )
        )
        # assert np.all(self.best_path == np.array(self.sp)), "paths differ"

    def _combined_paths(self, start, dest, best_shift, best_edge):
        """
        Compute path through one specific edge (with bi-directed predecessors)

        Arguments:
            start: overall source vertex
            dest: overall target vertex
            best_shift: the neighbor index of the edge
            best_edge: the vertex of the edge
        """
        # compute path from start to middle point - incoming edge
        best_edge = np.array(best_edge)
        path_ac = ut_ksp.get_sp_start_shift(
            self.dists, self.preds, self.pos2node, start, best_edge,
            np.array(self.shifts), best_shift
        )
        # compute path from middle point to dest - outgoing edge
        path_cb = ut_ksp.get_sp_dest_shift(
            self.dists_ba, self.preds_ba, self.pos2node, dest, best_edge,
            np.array(self.shifts) * (-1), best_shift
        )
        # concatenate
        together = np.concatenate(
            (np.flip(np.array(path_ac), axis=0), np.array(path_cb)[1:]),
            axis=0
        )
        return together

    # ---------------------------------------------------------------------
    # Functions to output path (backtrack) and corresponding costs

    def transform_path(self, path):
        # raw_resist = np.array([[self.instance[p[0], p[1]]] for p in path])

        # compute angle costs
        ang_costs = ut_cost.compute_angle_costs(
            path, self.angle_norm_factor, mode=self.angle_cost_function
        )
        # compute the cable costs (bresenham line between pylons)
        edge_costs = np.zeros(len(path))
        if self.edge_weight != 0:
            edge_costs = ut_cost.compute_edge_costs(path, self.edge_inst)

        # compute the geometric path costs
        path_costs = ut_cost.compute_geometric_costs(
            path, self.instance, edge_costs * self.edge_weight
        )
        # combine costs
        cost_sum = np.sum(path_costs) + self.angle_weight * np.sum(ang_costs)
        return np.asarray(path).tolist(), np.array(path_costs), cost_sum

    def get_shortest_path(self, start_inds, dest_inds, ret_only_path=False):
        dest_ind_stack = self.pos2node[tuple(dest_inds)]
        if not np.any(self.dists[dest_ind_stack, :] < np.inf):
            warnings.warn("empty path")
            return [], [], 0
        tic = time.time()
        curr_point = dest_inds
        path = [dest_inds]
        # first minimum: angles don't matter, just min of in-edges
        min_shift = np.argmin(self.dists[dest_ind_stack, :])
        # track back until start inds
        while np.any(curr_point - start_inds):
            new_point = curr_point - self.shifts[int(min_shift)]
            # get new shift from argmins
            curr_ind_stack = self.pos2node[tuple(curr_point)]
            min_shift = self.preds[curr_ind_stack, int(min_shift)]
            if min_shift == -1:
                print(path)
                raise RuntimeError("Problem! predecessor -1!")
            path.append(new_point)
            curr_point = new_point

        path = np.flip(np.asarray(path), axis=0)
        if ret_only_path:
            return path
        self.sp = path
        self.time_logs["path"] = round(time.time() - tic, 3)
        return self.transform_path(path)

    # ----------------------------------------------------------------------
    # Other auxiliary functions
    def save_graph(self, out_path):
        with open(out_path + ".dat", "wb") as outfile:
            pickle.dump((self.dists, self.preds), outfile)

    def _helper_list(self):
        tmp_list = List()
        tmp_list_inner = List()
        tmp_list_inner.append(0)
        tmp_list_inner.append(0)
        tmp_list.append(tmp_list_inner)
        return tmp_list

    def add_start_and_dest(self, source, dest):
        # here simply return the indices for start and destination
        return source, dest

    # -----------------------------------------------------------------------
    # INTERFACE

    def _check_start_dest(self, **kwargs):
        # assert that start and dest are provided and in project bounds
        try:
            self.start_inds = kwargs["start_inds"]
            self.dest_inds = kwargs["dest_inds"]
        except KeyError:
            raise RuntimeError(
                "Must specify start_inds and dest_inds in cfg dict!"
            )
        assert len(self.start_inds) == 2, "start inds must be of length 2"
        assert len(self.dest_inds) == 2, "dest inds must be of length 2"
        assert all(
            [
                isinstance(self.start_inds[i], int)
                or float(self.start_inds[i]).is_integer() for i in range(2)
            ]
        ), "start inds must be integer!"
        assert all(
            [
                isinstance(self.dest_inds[i], int)
                or float(self.dest_inds[i]).is_integer() for i in range(2)
            ]
        ), "dest inds must be integer!"
        assert self.instance[tuple(
            self.start_inds
        )] < np.inf, "Problem: Start coordinates are not in project region"
        assert self.instance[tuple(
            self.dest_inds
        )] < np.inf, "Problem: Target coordinates are not in project region"
        self.start_inds = np.asarray(self.start_inds).astype(int)
        self.dest_inds = np.asarray(self.dest_inds).astype(int)

        instance_shape = np.asarray(self.instance.shape)
        assert np.all(np.asarray(self.start_inds) < instance_shape) and np.all(
            np.asarray(self.dest_inds) < instance_shape
        ), "start or dest not in project region!"

    def single_sp(self, **kwargs):
        """
        Function for full processing to yield shortest path
        Necessary parameters:
            start_inds: list of two cell coordinates
            dest_inds: list of two cell coordinates
        Optional parameters:
            pylon_dist_min: min cell distance of neighboring pylons (default 3)
            pylon_dist_max: min cell distance of neighboring pylons (default 5)
            angle_weight: Importance of angle costs compared to resistances
                (=0 means only resistance is optimized, =1 means only angles
                are minimized, i.e. output will be straightest line possible)
            edge_weight: importance of cable costs vs pylon costs (default 0)
            max_angle: maximum deviation in angle from the straight connection
                       from start to end (default: pi/2)
            max_angle_lg: maximum angle at a pylon (default: pi/2)
        """
        # assert that start and dest exist in kwargs and are in project region
        self._check_start_dest(**kwargs)

        # initialize donut ring and edge costs
        self.set_shift(self.start_inds, self.dest_inds, **kwargs)
        if self.verbose:
            print("1) Initialize shifts and instance (corridor)")
        self.set_edge_costs(**kwargs)
        # add vertices
        self.add_nodes()
        if self.verbose:
            print("2) Initialize distances to inf and predecessors")
        # MAIN ALGORITHM
        self.build_source_sp_tree(**kwargs)
        if self.verbose:
            print("3) Compute source shortest path tree")
            print("number of vertices and edges:", self.n_nodes, self.n_edges)

        # get actual best path
        path, path_costs, cost_sum = self.get_shortest_path(
            self.start_inds, self.dest_inds
        )
        if self.verbose:
            print("4) shortest path", cost_sum)
        return path, path_costs, cost_sum

    def sp_trees(self, **kwargs):
        """
        Compute shortest path trees from both directions (Eppstein distances)
        necessary for finding multiple paths
        """
        # Build shortest path tree rooted in source
        path, path_costs, cost_sum = self.single_sp(**kwargs)
        # Build shortest path tree rooted in target
        self.build_dest_sp_tree(self.start_inds, self.dest_inds)
        return path, path_costs, cost_sum
