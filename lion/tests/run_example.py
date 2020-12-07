import numpy as np
import pickle
import time
import os
import argparse
import matplotlib.pyplot as plt

from lion.utils import general
from lion.algorithms import (
    optimal_point_spotting, optimal_route, ksp_points, ksp_routes
)


def plot_paths(plot_instance, paths, buffer=0, out_path="test_path.png"):

    # normalize instance
    plotmin = np.min(plot_instance[plot_instance < np.inf])
    plotmax = np.max(plot_instance[plot_instance < np.inf])
    instance = (plot_instance - plotmin) / (plotmax - plotmin)

    expanded = np.expand_dims(instance, axis=2)
    expanded = np.tile(expanded, (1, 1, 3))  # overwrite instance by tiled one
    # colour nodes in path in red
    for path in paths:
        for (x, y) in path:
            expanded[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                     1] = [0.9, 0.2, 0.2]  # colour red
    # plot and save
    plt.figure(figsize=(25, 25))
    plt.imshow(np.swapaxes(expanded, 1, 0))
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to pickled example data")
    parser.add_argument(
        "-s",
        "--save_path",
        default=None,
        type=str,
        help="Path where to save an output image and array of\
             the path if desired"
    )
    args = parser.parse_args()

    # load instance
    with open(args.path, "rb") as infile:
        data = pickle.load(infile)

    # get the array and config from the loaded data
    try:
        test_instance = data["matrix"]
        cfg = data["configs"][0]
    except (KeyError, ValueError) as e:
        raise e(
            "Wrong format of pickle file: must be a dictionary\
            with a key 'matrix' and a key 'configs' where matrix is a numpy\
            array and configs is a list with one dictionary inside"
        )

    print("shape of instance and cfg:", test_instance.shape, cfg)
    print("unique values in instance", np.unique(test_instance))

    # modify point distance (currently not provided)
    if "point_dist_min" not in cfg.keys():
        cfg["point_dist_min"] = 15
        cfg["point_dist_max"] = 25

    tic = time.time()
    path = optimal_point_spotting(test_instance, cfg.copy())
    paths = [path]
    # paths = ksp_pylons(test_instance, cfg.copy(), 5)
    print("overall time for processing", time.time() - tic)

    # save outputs
    if args.save_path is not None:
        # plotting - set high values to inf because otherwise everthing except
        # for the high cost area will be just black
        plot_instance = test_instance.copy().astype(float)
        plot_instance[plot_instance == np.max(plot_instance)] = np.inf
        # set buffer: Bigger instance --> use bigger buffer
        plot_paths(
            plot_instance,
            paths,
            buffer=2,
            out_path=os.path.join(
                os.path.join(args.save_path, "test_output.png")
            )
        )
        with open(os.path.join(args.save_path, "paths.dat"), "wb") as outfile:
            pickle.dump(paths, outfile)
