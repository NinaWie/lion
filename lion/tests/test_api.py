import unittest
import numpy as np
from lion.utils.plotting import plot_paths
from lion.algorithms import (
    optimal_point_spotting, optimal_route, ksp_points, ksp_routes
)


class TestAPI(unittest.TestCase):

    test_instance = np.random.rand(100, 100)
    num_nans = 50
    forb_x = (np.random.rand(num_nans) * 100).astype(int)
    forb_y = (np.random.rand(num_nans) * 100).astype(int)
    test_instance[forb_x, forb_y] = np.nan

    # create configuration
    cfg = dict()
    cfg["start_inds"] = [6, 6]
    cfg["dest_inds"] = [94, 90]
    test_instance[tuple(cfg["start_inds"])] = np.random.rand(1)[0]
    test_instance[tuple(cfg["dest_inds"])] = np.random.rand(1)[0]

    def test_optimal_route(self) -> None:
        path = optimal_route(
            self.test_instance,
            self.cfg.copy(),
        )
        self.assertTrue(len(path) > 0)
        self.assertListEqual(list(path[0]), list(self.cfg["start_inds"]))
        self.assertListEqual(list(path[-1]), list(self.cfg["dest_inds"]))

        plot_paths(
            self.test_instance, [path],
            buffer=0,
            out_path="test_optimal_route.png"
        )

    def test_optimal_point_spotting(self) -> None:
        path = optimal_point_spotting(
            self.test_instance,
            self.cfg.copy(),
        )
        self.assertTrue(len(path) > 0)
        self.assertListEqual(list(path[0]), list(self.cfg["start_inds"]))
        self.assertListEqual(list(path[-1]), list(self.cfg["dest_inds"]))

        plot_paths(
            self.test_instance, [path],
            buffer=0,
            out_path="test_optimal_point_spotting.png"
        )
        cfg = self.cfg.copy()
        cfg["pipeline"] = [3, 1]
        path_pipeline = optimal_point_spotting(
            self.test_instance,
            cfg,
        )
        self.assertListEqual(path, path_pipeline)

    def test_ksp_routes(self) -> None:
        paths = ksp_routes(self.test_instance, self.cfg.copy(), 5)
        self.assertTrue(len(paths) == 5)
        plot_paths(self.test_instance, paths, out_path="test_route_ksp.png")

    def test_ksp_points(self) -> None:
        paths = ksp_points(self.test_instance, self.cfg.copy(), 5)
        self.assertTrue(len(paths) == 5)
        plot_paths(self.test_instance, paths, out_path="test_point_ksp.png")


if __name__ == '__main__':
    unittest.main()
