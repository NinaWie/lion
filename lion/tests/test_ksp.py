import unittest
import numpy as np
from lion.utils.plotting import plot_paths
from lion.algorithms import (ksp_points, ksp_routes)
from lion.ksp import KSP


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
    cfg["point_dist_min"] = 5
    cfg["point_dist_max"] = 7
    test_instance[tuple(cfg["start_inds"])] = np.random.rand(1)[0]
    test_instance[tuple(cfg["dest_inds"])] = np.random.rand(1)[0]

    def test_euclidean(self) -> None:
        thresh = 10
        self.cfg["thresh"] = thresh
        paths = ksp_points(
            self.test_instance, self.cfg.copy(), 5, algorithm=KSP.ksp
        )
        self.assertTrue(len(paths) == 5)
        for i in range(5):
            for j in range(0, i):
                path1 = paths[i]
                path2 = paths[j]
                min_dists_out = []
                for p1 in range(len(path1)):
                    min_dists = []
                    for p2 in range(len(path2)):
                        min_dists.append(np.linalg.norm(path1[p1] - path2[p2]))
                    min_dists_out.append(np.min(min_dists))
                self.assertGreaterEqual(np.max(min_dists_out), thresh)

    def test_min_intersection(self) -> None:
        thresh = .5
        k = 4
        self.cfg["thresh"] = thresh
        paths = ksp_points(
            self.test_instance,
            self.cfg.copy(),
            k,
            algorithm=KSP.min_set_intersection
        )
        k = len(paths)

        # check min intersection
        for i in range(k - 1):
            for j in range(i + 1, k):
                # compute pairwise intersections
                path1 = paths[i]
                path2 = paths[j]
                max_point = max([np.max(path1), np.max(path2)])
                path_1_transformed = path1[:, 0] + max_point * path1[:, 1]
                path_2_transformed = path2[:, 0] + max_point * path2[:, 1]
                intersection = np.intersect1d(
                    path_1_transformed, path_2_transformed
                )
                max_length = max([len(path1), len(path2)])
                # print(len(intersection), len(path1), len(path2))

                # assert intersection is small enought
                self.assertLessEqual(len(intersection) / max_length, thresh)


if __name__ == '__main__':
    unittest.main()
