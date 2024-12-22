from minitorch._tensor import Tensor
import minitorch.core.operators as operators
import random

from typing import Any, TYPE_CHECKING
from minitorch._tensor_helpers import UserShape

if TYPE_CHECKING:
    pass


# Helpers for Constructing tensors
def zeros(shape: UserShape, device) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        device : tensor backend

    Returns:
        new tensor

    """
    return Tensor.make([0] * int(operators.prod(shape)), shape, device=device)


def rand(
    shape: UserShape,
    device,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        device : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = Tensor.make(vals, shape, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    device,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        device: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor

    """
    tensor = Tensor.make(ls, shape, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(ls: Any, device, requires_grad: bool = False) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        device : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> list[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> list[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), device=device, requires_grad=requires_grad)
