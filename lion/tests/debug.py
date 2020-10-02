import numpy as np

from lion.algorithms import (
    optimal_pylon_spotting, optimal_route, ksp_pylons, ksp_routes
)

import matplotlib.pyplot as plt  # TODO


def plot_paths(instance, paths, buffer=0, out_path="test_path.png"):
    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for path in paths:
        for (x, y) in path:
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 0.2, 0.2]  # colour red
    # plot and save
    plt.figure(figsize=(25, 15))
    plt.imshow(np.swapaxes(expanded, 1, 0))
    plt.savefig(out_path, bbox_inches='tight')


test_instance = np.random.rand(100, 100)
num_nans = 100
forb_x = (np.random.rand(num_nans) * 100).astype(int)
forb_y = (np.random.rand(num_nans) * 100).astype(int)
test_instance[forb_x, forb_y] = np.nan

# create configuration
cfg = dict()
cfg["start_inds"] = [6, 6]
cfg["dest_inds"] = [94, 90]
test_instance[tuple(cfg["start_inds"])] = 0
test_instance[tuple(cfg["dest_inds"])] = 0

test_instance = np.load("debug.npy")

cfg = {
    'start_inds': [2, 2],
    'dest_inds': [37, 98],
    'angle_weight': 0,
    'max_angle': 2.0707963267948966,
    'pylon_dist_min': 0.9,
    'pylon_dist_max': 1.5
}

path = optimal_route(
    test_instance,
    cfg.copy(),
)
print(path)
plot_paths(test_instance, [path], out_path="test.png")
