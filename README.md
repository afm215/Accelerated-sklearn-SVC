# Accelerated-SVC
## Description

Repository for accelerated class wrappers for sklearn SVC.
The wrapper computes the training and inference sample cross similarities using PyTorch, which parallelizes the computations on either CPU or GPU.
This can significantly decrease the computation time, especially when working with many vectors and/or vectors in high dimensions. 


## Required packages
An uv pyproject will be added, but the following packages are required:
- torch
- numpy
- tqdm
- scikit-learn

## Supported kernel
- rbf
- mahalanobis
- linear

## TODO
- Add unit tests to ensure results quality, reporting the gains in terms of time 
- Add polynomial kernels
- Add a pyproject.toml file (OPTIONAL since it should work with most package versions) 
