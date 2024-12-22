"""Contains the base class for tensor operations and a slow implementation"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Type

from typing_extensions import Protocol

from minitorch.core import operators

if TYPE_CHECKING:
    from ._tensor import Tensor


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        pass


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:  # type:ignore[empty-body]
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type:ignore[empty-body]
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:  # type:ignore[empty-body]
        pass

    @staticmethod
    def reduce(  # type:ignore[empty-body]
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.tanh_map = ops.map(operators.tanh)
        self.gelu_map = ops.map(operators.gelu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.tanh_back_zip = ops.zip(operators.tanh_back)
        self.gelu_back_zip = ops.zip(operators.gelu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.max_reduce = ops.reduce(operators.max, float("-inf"))
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda
