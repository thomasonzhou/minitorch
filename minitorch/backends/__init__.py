from minitorch._ops import TensorBackend
from .simple_ops import SimpleOps
from .fast_ops import FastOps
from .cuda_ops import CudaOps

from enum import Enum
from typing import Callable, Any
from functools import wraps


class Backend(Enum):
    SIMPLE = TensorBackend(SimpleOps)
    FAST = TensorBackend(FastOps)
    CUDA = TensorBackend(CudaOps)


DEFAULT_BACKEND = Backend.FAST
_current_backend = Backend.FAST


def get_backend() -> TensorBackend:
    return _current_backend.value


def set_backend(b: TensorBackend):
    global _current_backend
    _current_backend = b


def reset_backend():
    global _current_backend
    _current_backend = DEFAULT_BACKEND


def with_current_backend(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, device: TensorBackend = None, **kwargs) -> Any:
        if device is None:
            device = get_backend()
        return func(*args, **kwargs, device=device)

    return wrapper
