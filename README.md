# About 

This is an reimplementation of the PyTorch API. It supports the following:

- [x] Autodifferentiation / Backpropagation
- [x] Tensors, Views, Strides
- [x] GPU / CUDA programming in Numba
  - [x] Map / Zip / Reduce
  - [x] Batched matrix multiplication
- [x] 1D / 2D Convolution and Pooling

# Getting started

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

# Further extensions
- [ ] Saving and loading torch state dictionaries
- [ ] Transformer module
  - [ ] tanh, gelu
- [ ] Embedding module
- [ ] Expand core tensor operations
  - [ ] arange, cat, stack, hstack
- [ ] ADAM optimizer
- [ ] Additional loss functions

# Credit

Building this would have been impossible without the original course:
[Minitorch by Sasha Rush](https://minitorch.github.io/)
