from .tensor_ops import TensorBackend
from .fast_ops import FastOps

DEFAULT_BACKEND = TensorBackend(FastOps) 
