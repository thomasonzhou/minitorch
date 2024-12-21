from minitorch._ops import TensorBackend
from .cuda_ops import CudaOps
from .fast_ops import FastOps

cuda_backend = TensorBackend(CudaOps)
fast_backend = TensorBackend(FastOps)


def get_default_backend():
    return fast_backend
