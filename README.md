# Linear infrastructure optimization networks (LION)



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
 jgs \         /_ | |__
     (___________)))))))





Given resistance costs for a raster of geo locations, the goal is to compute the optimal power line layout from a given start point to a given end point. The approach is to represent raster cells as vertices in a graph, place edges between them based on the minimal and maximal distance of the power towers, and define edge costs based on the underlying cost surface.

## Installation

The library itself has few major dependencies (see [setup.py](setup.py)). 
* `numba`for fast algorithms
* `numpy`
* `scipy`

If wanted, create a virtual environment and activate it:

```sh
python3 -m venv env
source env/bin/activate
```

Install the repository in editable mode:

```sh
git clone https://gitlab.eth.ch/wnina/lion
cd lion
pip install -e .
```

## Optimal power infrastructure planning

All main functions are available in [algorithms](lion/algorithms.py). Usage is examplified in [test_api](lion/tests/test_api.py).



