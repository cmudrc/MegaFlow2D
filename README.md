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

## Dataset structure
The entire dataset is stored inside a single HDF5 file. Although multiple HDF5 files are created during processing depending on the number of processing cores used to avoid data corruption while concurrently writing to a single file. The reading operation, however, can be done concurrently as long as all operations are restricted in `r` mode. The dataset is stored in a hierarchical structure, and each group is indexed by the geometry type, mesh resolution and time step. The dataset object is stored as a `h5py.dataset` object under each group. The dataset structure is shown below:
```bash
├── MegaFlow2D
│   ├── <geometry_type>_<geometry_index>
│   │   ├── <mesh_resolution>
│   │   │   ├── <time_step>
│   │   │   │   ├── dataset

```
In theory, searching through the dataset can have a complexity of O(1) due to the B-tree structure of HDF5 to allow for fast data retrieval in training loading process. However, the process might be slowed down by the auto decompression of the dataset. This may be improved by reprocessing the dataset with a different compression setting in `utils.py`. Please keep in mind that reprocessing the dataset can take several hours depending on the number of cores used.

## Using the MegaFlow package

The MegaFlow package provides a simple interface for initializing and loading the dataset. 

```py
from megaflow.dataset.MegaFlow2D import MegaFlow2D

if __name__ == '__main__':
    dataset = MegaFlow2D(root='/path/to/your/directory', download=True, transform='normalize', pre_transform=None, split_scheme='mixed', split_ratio=0.8)
    # if the dataset is not processed, the process function will be called automatically. 
    # to facilitate multi-thread processing, be sure to exceute the process function in '__main__'.

    # get one sample
    sample_low, sample_high = dataset.get(0)
    print('Number of nodes: {}, number of edges: {}'.format(sample_low.num_nodes, sample_low.num_edges))
```

## Using the example scripts
We provide an example script for training a super-resolution model on the MegaFlow2D dataset. The script can be found in the `examples` directory. The script can be run by (one configuration example):
```bash
python examples/train.py --root /path/to/your/directory --dataset MegaFlow2D --tranform normalize --model FlowMLError --epochs 100 --batch_size 32 
```

## Citing MegaFlow2D
If you use MegaFlow2D in your research, please cite:
```bibtex
@InProceedings{xu2023,
  author={Xu, Wenzhuo and Grande Gutiérrez, Noelia and McComb, Christopher},
  booktitle={2023 IEEE Workshop on Design Automation for CPS and IoT (DESTION)}, 
  title={MegaFlow2D: A Parametric Dataset for Machine Learning Super-resolution in Computational Fluid Dynamics Simulations}, 
  year={2023},
  pages={TBD},
  doi={TBD}
}
```
