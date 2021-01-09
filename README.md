![build](https://github.com/NinaWie/lion/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/NinaWie/lion/branch/master/graph/badge.svg?token=e9c953ec-6da4-4729-8dfa-636c8638e6df)](https://codecov.io/gh/NinaWie/lion)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# LInear Optimization Networks (LION)


    
                 ,%%%%%%%%,
               ,%%/\%%%%/\%%
              ,%%%\c "" J/%%%
     %.       %%%%/ o  o \%%%
     `%%.     %%%%    _  |%%%
      `%%     `%%%%(__Y__)%%'
      //       ;%%%%`\-/%%%'
     ((       /  `%%%%%%%'
      \\    .'          |
       \\  /       \  | |
        \\/         ) | |
         \         /_ | |__
         (___________)))))))
    

This package implements variants of shortest path algorithms to compute **minimal angle** shortest paths and **k diverse shortest paths** through a cost / resistance array.

Possible applications include 
* Computer vision tasks
* Route finding in robotics
* Infrastructure planning

## Installation

The library itself has few major dependencies (see [setup.py](setup.py)). 
* `numba` (for fast compiled algorithms)
* `numpy`
* `scipy`



**Installation from git:**

Create a virtual environment and activate it:

```sh
python3 -m venv env
source env/bin/activate
```

Install the repository in editable mode:

```sh
git clone https://github.com/NinaWie/lion
cd lion
pip install -e .
```

## Instructions

All main functions are available in [algorithms](lion/algorithms.py). Usage is examplified in [test_api](lion/tests/test_api.py).

**Input:** 

* `instance`: 2D numpy array of real float or integer values which are the costs
* configuration dictionary with 
  * `start_inds`: start coordinates, e.g. [0,0]
  * `dest_inds`: target coordinates, e.g. [20,20]
  * `angle_weight`: Float between 0 and 1, 0: angles don't matter, 1: only angles are minimized, costs are not taken into account
  * `forbidden_val`: value indicating that a cell is forbidden / outside of the project region (can be int, np.nan, np.inf ... )
  * `point_dist_min`: minimum cell distance of neighboring points (default 3)
  * `point_dist_max`: minimum cell distance of neighboring points (default 5)
  * `angle_weight`: how important is the angle (default 0)
  * `edge_weight`: importantance of costs between points compared to the cost at the points (default=0, i.e. only the points count)
  * `max_angle`: maximum deviation in angle from the straight connection from start to end (default: pi/2)
  * `max_angle_lg`: maximum angle of adacent edges (default: pi, i.e. no restriction)
  * `angle_cost_function`: 'linear' and 'discrete' are implemented. See [code](lion/utils/general.py)
  * `memory_limit`: default is 1 trillion, if the number of edges is higher, an iterative approximation procedure (''pipeline'') is used
  * `pipeline`: List of decreasing positive integers, ending with 1
            The pipeline in an iterative approach defines the downsampling
            factors for each step. By default, it is set automatically based on
            the memory limit. It can however be set manually as well, e.g.
            [3,1] means downsample by factor of 3, compute optimal
            path, reduce region of interest to a corridor around
            optimal path (corridor width is computed automatically based on the
            memory_limit) then downsample by factor of 1 (aka full resolution).
            There is no support for setting the corridor width manually because
            it does not make sense to make it smaller than it could be
  * `between_points_allowed`: If True, then forbidden areas can still be between points (only relevant for point routing)
            If False, then it is not allowed that a forbidden area is between two points of the path
  * `diversity_threshold`:
    * FOR KSP.ksp:
            Minimum diversity of next path from previous paths in cell
            distance. E.g. if thresh = 200, each path will be at least 200
            cells away at one point from each other path.
            If None, it is set by default to 1/20 of the instance size
    * FOR KSP.min_set_intersection:
            maximum intersection of the new path with all previous points
            Must be between 0 and 1. E.g. if 0.2, then at most 20% of cells
            are shared between one path and the other paths

## Example usage

```sh
np.random.seed(0)
instance = np.random.rand(100,100)
# set some forbidden values
instance[instance>0.9] = np.inf
# define config dictionary
cfg = {
  "start_inds": [0,2],
  "dest_inds": [96, 93],
  "angle_weight": 0.5,
  "forbidden_val": np.inf
}

# Compute the least cost route:
opt_route = optimal_route(instance, cfg)
# >>> optimal_route.shape
# (120, 2) --> x and y coordinates of 120 cells on the route

# compute k diverse routes:
multiple_routes = ksp_routes(instance, cfg, 5)
# >>> len(multiple_routes)
# 5
# >>> np.all(np.array(multiple_routes[0]) == np.array(opt_route)))
# True --> 1st route is the optimal one, other routes are diverse alternatives 

# define minimum and maximum distance between points
cfg["point_dist_min"] = 5
cfg["point_dist_max"] = 7

opt_points = optimal_points(instance, cfg)
# >>> len(opt_points)
# 26 --> only 26 because they can be more than 1 apart
# >>> np.linalg.norm(np.array(opt_points[0]) - np.array(opt_points[1]))
# 5.83095189 --> Euclidean distance between cells is between 5 and 7

multiple_points = ksp_points(instance, cfg, 5)
# >>> len(multiple_points)
# 5
```
