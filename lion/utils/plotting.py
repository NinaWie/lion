import numpy as np
import logging

try:
    import matplotlib.pyplot as plt
    _PLOT_ENABLED = True
except ImportError:
    _PLOT_ENABLED = False

logger = logging.getLogger(__name__)


def angle_graph_display_dists(self, edge_array, func=np.min, name="dists"):
    if not _PLOT_ENABLED:
        logger.warning('No plotting library available')
        return

    arr = np.zeros(self.pos2node.shape)
    for i in range(len(self.pos2node)):
        for j in range(len(self.pos2node[0])):
            ind = self.pos2node[i, j]
            if ind >= 0:
                arr[i, j] = func(edge_array[ind, :])
    plt.imshow(arr)
    plt.colorbar()
    plt.savefig(name + ".png")


def plot_paths(instance, paths, buffer=0, out_path="test_path.png"):
    if not _PLOT_ENABLED:
        logger.warning('No plotting library available')
        return
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


def plot_dists(self, dists, pos2node, save_name):
    arr = np.zeros(pos2node.shape)
    for i in range(len(pos2node)):
        for j in range(len(pos2node)):
            if pos2node[i, j] >= 0:
                arr[i, j] = np.min(dists[pos2node[i, j]])
    plt.imshow(arr)
    plt.savefig(save_name)
