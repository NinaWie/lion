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

    arr = np.zeros(self.pos2node.shape)
    for i in range(len(self.pos2node)):
        for j in range(len(self.pos2node[0])):
            ind = self.pos2node[i, j]
            if ind >= 0:
                arr[i, j] = func(edge_array[ind, :])
    plt.imshow(arr)
    plt.colorbar()
    plt.savefig(name + ".png")
