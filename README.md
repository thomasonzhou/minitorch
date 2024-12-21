# About 

This is an reimplementation of the PyTorch API. It supports the following:

- [x] Autodifferentiation / Backpropagation
- [x] Tensors, Views, Strides
- [x] GPU / CUDA programming in Numba
  - [x] Map / Zip / Reduce
  - [x] Batched matrix multiplication
- [x] 1D / 2D Convolution and Pooling

# Getting started

## Using [uv](https://github.com/astral-sh/uv)

To install dependencies, create a virtual environment and install the required packages:
```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If pip raises an error, it may be necessary to upgrade with:
```python
pip install --upgrade pip
```
This will also install minitorch in editable mode.

## Examples
### Training a MNIST model

```python
python project/run_mnist_multiclass.py 
```

# Credit

Building this would have been impossible without the original course:
[Minitorch by Sasha Rush](https://minitorch.github.io/)
