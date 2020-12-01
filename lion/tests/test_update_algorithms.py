import unittest
import numpy as np
from lion.angle_graph import AngleGraph
from types import SimpleNamespace
from lion.fast_shortest_path import sp_dag
from lion.utils.shortest_path import (
    update_default, update_linear, update_discrete
)


class TestUpdateAlgs(unittest.TestCase):

    expl_shape = (100, 100)
    instance = np.random.rand(*expl_shape)
    corridor = (np.random.rand(*expl_shape) > 0.05).astype(int)

    # create configuration
    cfg = dict()
    start_inds = np.array([6, 6])
    dest_inds = np.array([94, 96])
    corridor[tuple(start_inds)] = 1
    corridor[tuple(dest_inds)] = 1
    cfg["start_inds"] = start_inds
    cfg["dest_inds"] = dest_inds
    cfg["point_dist_min"] = 10
    cfg["point_dist_max"] = 15
    cfg["layer_classes"] = ["dummy_class"]
    cfg["class_weights"] = [1]

    def test_linear(self) -> None:
        """ LINEAR """
        self.cfg["angle_cost_function"] = "linear"
        self.cfg["max_angle_lg"] = np.pi
        graph = AngleGraph(self.instance, self.corridor)
        _ = graph.single_sp(**self.cfg)
        gt_angle_cost_arr = graph.angle_cost_array.copy()
        gt_dists = graph.dists.copy()
        graph.dists = np.zeros(
            (len(graph.stack_array), len(graph.shifts))
        ) + np.inf
        graph.dists[0, :] = 0
        graph.preds = np.zeros(graph.dists.shape) - 1
        graph.angle_cost_function = "some_non_existant"
        graph.build_source_sp_tree(**self.cfg)

        self.assertTrue(
            np.all(np.isclose(gt_angle_cost_arr, graph.angle_cost_array))
        )
        self.assertTrue(np.all(np.isclose(gt_dists, graph.dists)))

        # OTHER WAY: FROM NOTEBOOK
        # stack_saved = graph.stack_array.copy()
        # self.assertTrue(np.all(graph.angle_cost_array < np.inf))
        # dists_new = np.zeros(
        #     (len(graph.stack_array), len(graph.shifts))
        # ) + np.inf
        # dists_new[0, :] = 0
        # preds_new = np.zeros(dists_new.shape) - 1

        # gt_dists, _ = sp_dag(
        #     stack_saved, graph.pos2node,
        #     np.array(graph.shifts), graph.angle_cost_array, dists_new.copy(),
        #     preds_new.copy(), graph.edge_cost, update_default,
        #     (graph.angle_cost_array)
        # )

        # linear_dists, _ = sp_dag(
        #     stack_saved, graph.pos2node,
        #     np.array(graph.shifts), graph.angle_cost_array, dists_new.copy(),
        #     preds_new.copy(), graph.edge_cost, update_linear,
        #     (graph.angle_cost_array)
        # )

        # self.assertTrue(np.all(gt_dists == linear_dists))

    def test_discrete(self) -> None:
        """ DISCRETE """
        self.cfg["angle_cost_function"] = "discrete"
        self.cfg["max_angle_lg"] = np.pi / 4
        graph = AngleGraph(self.instance, self.corridor)
        _ = graph.single_sp(**self.cfg)
        gt_angle_cost_arr = graph.angle_cost_array.copy()
        gt_dists = graph.dists.copy()
        graph.dists = np.zeros(
            (len(graph.stack_array), len(graph.shifts))
        ) + np.inf
        graph.dists[0, :] = 0
        graph.preds = np.zeros(graph.dists.shape) - 1
        graph.angle_cost_function = "some_non_existant"
        graph.build_source_sp_tree(**self.cfg)

        self.assertTrue(np.all(gt_angle_cost_arr == graph.angle_cost_array))
        self.assertTrue(np.all(np.isclose(gt_dists, graph.dists)))


if __name__ == '__main__':
    unittest.main()
