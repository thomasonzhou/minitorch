# PyTorch from Numpy and Numba

This is an reimplementation of a subset of the torch API. It supports the following:

- [x] autodifferentiation / backpropagation
- [x] tensors, views, broadcasting
- [x] GPU / CUDA programming in Numba
  - [x] map / zip / reduce
  - [x] batched matrix multiplication
- [x] 1D / 2D Convolution and Pooling
- [x] activation functions
  - [x] ReLU / GeLU / softmax / tanh
- [x] optimizers
  - [x] stochastic gradient descent

# Getting Started

To install dependencies, create a virtual environment and install the required packages:
```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
This will install minitorch in editable mode.

If pip raises an error, it may be necessary to upgrade before installing dependencies:
```python
pip install --upgrade pip
```
## Examples
### Training a MNIST model

```python
python project/run_mnist_multiclass.py 
```
### Creating a custom model
A list of supported modules and functions are listed in examples/custom.py.

# Repo Structure

Files prefixed with a leading underscore implement abstract base classes and tensor manipulation functions.

| Subpackage | Description |
|------------|-------------|
| autograd | central difference / topological sort of computational graph |
| backends | naive / parallel / CUDA implementations of map / zip / reduce / matrix multiply |
| nn | modules and functions for building networks |
| optim | optimizers for loss function minimization |

# Extensions

## Features
- [ ] Saving and loading 
  - [ ] torch state dictionaries
  - [ ] ONNX
- [ ] Transformer module
  - [x] tanh, GeLU
- [ ] Embedding module
- [ ] Expand core tensor operations
  - [ ] arange, cat, stack, hstack
- [ ] Adam optimizer
- [ ] Additional loss functions
- [ ] Einsum!

## Optimizations
- [ ] Bindings
- [ ] CUDA Convolution

## Documentation
- [ ] CUDA usage with Google Collab

# Credit

Building this would have been impossible without the original course:
[Minitorch by Sasha Rush](https://minitorch.github.io/)
