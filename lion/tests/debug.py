import numpy as np
import pickle
from lion.utils import general
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


# GILYTICS Instance
with open("data/rmap_ndarray_11.pickle", "rb") as infile:
    test_instance = pickle.load(infile)

sp = np.asarray([962, 122])
ep = np.asarray([585, 2601])
# downsample for fast testing
factor = 1
if factor > 1:
    test_instance = general.rescale(test_instance, factor)
    sp = (sp / factor).astype(int)
    ep = (ep / factor).astype(int)
cfg = {}
cfg["start_inds"] = sp
cfg["dest_inds"] = ep
cfg["angle_weight"] = 0
print(test_instance.shape, cfg)

# path = optimal_route(test_instance, cfg.copy())
# paths = [path]
paths = ksp_routes(test_instance, cfg.copy(), 5)

# plotting
print(len(paths))
plot_instance = test_instance.copy().astype(float)
plot_instance[plot_instance == 28000] = np.inf
plotmin = np.min(plot_instance[plot_instance < np.inf])
plotmax = np.max(plot_instance[plot_instance < np.inf])
plot_instance = (plot_instance - plotmin) / (plotmax - plotmin)
plot_paths(plot_instance, paths, out_path="test.png")
