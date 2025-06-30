# Accelerated-SVC
## Descrption

Repository for accelerated class wrappers for sklearn SVC.
The wrapper compputes the training samples cross similarities using torch, which apralelizes the computations either on cpu or gpu.
This can make the computation time de crease a lot, especially when working with many vectors and/or vectors in high dimensions. 


## Required packages
An uv pyproject will be added but the following packages are require:
- torch
- numpy
- tqdm
- scikit-learn

## Supported kernel
- rbf
- mahalanobis
- linear

## TODO
- Add unit test to ensure results quality
- Add polynomial kernel 
- Add pyproject.toml file (OPTIONAL since it should work with most packages versions) 
