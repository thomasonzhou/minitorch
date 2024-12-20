# About 

This is an reimplementation of the PyTorch API. It supports the following:

- [x] Automatic differentiation
- [x] Backpropagation
- [x] Tensors
- [x] CUDA backend

# Getting started

To install dependencies, create a virtual environment and install the required packages:

(I recommend uv because it is fast)
```python
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements.extra.txt
uv pip install -Ue .
```

