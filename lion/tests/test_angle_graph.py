import numpy as np
import unittest
from types import SimpleNamespace
from lion.angle_graph import AngleGraph
from lion.utils.general import bresenham_line
from lion.utils.costs import compute_geometric_costs


class TestImplicitLG(unittest.TestCase):

    expl_shape = (50, 50)

    # create configuration
    cfg = SimpleNamespace()
    cfg.point_dist_min = 3
    cfg.point_dist_max = 5
    start_inds = np.array([6, 6])
    dest_inds = np.array([41, 43])
    cfg.start_inds = start_inds
    cfg.dest_inds = dest_inds
    cfg.angle_weight = 0.25
    cfg.edge_weight = 0
    cfg.max_direction_deviation = np.pi / 2
    cfg.max_angle = np.pi / 4
    cfg.layer_classes = ["dummy_class"]
    cfg.class_weights = [1]

    # construct simple line instance
    example_inst = np.ones(expl_shape)
    # construct corresponding corridor
    working_expl_corr = np.zeros(expl_shape)
    line = bresenham_line(
        start_inds[0], start_inds[1], dest_inds[0], dest_inds[1]
    )
    for (i, j) in line:
        working_expl_corr[i - 2:i + 2, j - 2:j + 2] = 1

    # construct instance that required 90 degree angle
    high_angle_corr = np.zeros(expl_shape)
    high_angle_corr[start_inds[0], start_inds[1]:dest_inds[1] - 4] = 1
    high_angle_corr[start_inds[0] + 4:dest_inds[0] + 1, dest_inds[1]] = 1
    # no_connection_corr: there is no way to get from start to dest at all
    # with given point_dist_max
    no_connection_corr = high_angle_corr.copy()
    # high_angle_corr: there is a connection, but it requires a sharp turn
    # and is not possible with smaller max_angle
    high_angle_corr[start_inds[0], dest_inds[1]] = 1

    def test_correct_shortest_path(self) -> None:
        """ Test the implicit line graph construction """
        graph = AngleGraph(self.example_inst, self.working_expl_corr)
        path = graph.single_sp(**vars(self.cfg))
        path, path_costs, cost_sum = graph.transform_path(path)
        self.assertTupleEqual(graph.instance.shape, self.expl_shape)
        self.assertEqual(graph.angle_weight + graph.resistance_weight, 1)
        self.assertNotEqual(len(graph.shifts), 0)
        # all initialized to infinity
        # self.assertTrue(not np.any([graph.dists[graph.dists < np.inf]]))
        # start point was set to normalized value
        start_ind = graph.pos2node[tuple(self.start_inds)]
        self.assertEqual(0, graph.dists[start_ind, 0])
        # all start dists have same value
        self.assertEqual(len(np.unique(graph.dists[start_ind])), 1)
        # not all values still inf
        self.assertLess(np.min(graph.dists[start_ind]), np.inf)
        # get actual best path
        # path, path_costs, cost_sum = graph.get_shortest_path(
        #     self.start_inds, self.dest_inds
        # )
        self.assertNotEqual(len(path), 0)
        self.assertEqual(len(path), len(path_costs))
        self.assertGreaterEqual(cost_sum, 5)
        for (i, j) in path:
            self.assertEqual(self.example_inst[i, j], 1)

    def test_edge_costs(self) -> None:
        graph = AngleGraph(self.example_inst, self.working_expl_corr)
        self.cfg.angle_weight = 0
        self.cfg.edge_weight = 0.5
        path = graph.single_sp(**vars(self.cfg))
        path, _, cost_sum = graph.transform_path(path)
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        dest_costs = np.min(graph.dists[dest_ind])
        a = []
        path = np.array(path)
        weighted_inst = self.example_inst
        for p in range(len(path) - 1):
            line = bresenham_line(
                path[p, 0], path[p, 1], path[p + 1, 0], path[p + 1, 1]
            )[1:-1]
            line_costs = [weighted_inst[i, j] for (i, j) in line]
            line_costs = [li for li in line_costs if li < np.inf]
            bresenham_edge_dist = np.mean(line_costs) * self.cfg.edge_weight
            shift_costs = np.linalg.norm(path[p] - path[p + 1])
            # append edge cost
            a.append(
                (
                    0.5 * (
                        weighted_inst[tuple(path[p])] +
                        weighted_inst[tuple(path[p + 1])]
                    ) + shift_costs * bresenham_edge_dist
                )
            )
        dest_costs_gt = np.sum(a)
        self.assertTrue(np.isclose(cost_sum, dest_costs_gt))
        self.assertTrue(np.isclose(dest_costs, dest_costs_gt))
        out_compute_cost_method = compute_geometric_costs(
            path, weighted_inst, edge_weight=self.cfg.edge_weight
        )
        self.assertEqual(np.sum(out_compute_cost_method), cost_sum)

        self.cfg.angle_weight = 0.25
        self.cfg.edge_weight = 0

    def test_angle_sp(self) -> None:
        graph = AngleGraph(self.example_inst, self.high_angle_corr)
        _ = graph.single_sp(**vars(self.cfg))
        # assert that destination can NOT be reached
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        self.assertFalse(np.min(graph.dists[dest_ind]) < np.inf)
        # Note: this test does not change the stack array because the path is
        # only empty because of an inf in angles_all
        # NEXT TRY: more angles allowed
        self.cfg.max_angle = np.pi
        graph = AngleGraph(self.example_inst, self.high_angle_corr)
        path = graph.single_sp(**vars(self.cfg))
        path, _, cost_sum = graph.transform_path(path)
        # assert that dest CAN be reached
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        self.assertTrue(np.min(graph.dists[dest_ind]) < np.inf)

    def test_empty_path(self) -> None:
        self.cfg.max_angle = np.pi
        graph = AngleGraph(self.example_inst, self.no_connection_corr)
        _ = graph.single_sp(**vars(self.cfg))
        # assert that destination can NOT be reached
        dest_ind = graph.pos2node[tuple(self.dest_inds)]
        self.assertFalse(np.min(graph.dists[dest_ind]) < np.inf)


if __name__ == '__main__':
    unittest.main()
