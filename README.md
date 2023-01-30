# MegaFlow2D
 

## Overview
The MegaFlow2D dataset package of parameteric CFD simulation results for machine learning / super-resolution purposes.

The package contains:
1. A standard structure for transferring simulation results into graph structure.
2. Common utility functions for visualizing, retrieving and processing simulation results. (Everything that requires the [FEniCS](https://fenicsproject.org/) or [dolfin](https://github.com/FEniCS/dolfinx) package can only be run on linux or wsl.)

## Installation
The MegaFlow dataset can be installed by pip:
```bash
pip install MegaFlow2D
```

Running `pip install` would automatically configure package dependencies, however to build graphical models [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) needs to be installed manually.

## Using the MegaFlow package

The MegaFlow package provides a simple interface for initializing and loading the dataset. 

```py
from megaflow.dataset import MegaFlow2D

dataset = MegaFlow2D(root='/path/to/your/directory', download=True)

# get one sample
sample = dataset.get(0)
print('Number of nodes: {}, number of edges: {}'.format(sample.num_nodes, sample.num_edges))
```

## Using the example scripts
We provide an example script for training a super-resolution model on the MegaFlow2D dataset. The script can be found in the `examples` directory. The script can be run by (one configuration example):
```bash
python examples/train.py --root /path/to/your/directory --dataset MegaFlow2D --model FlowMLError --epochs 100 --batch_size 32 
```