from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias


MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
        index : index tuple of ints
        strides : tensor strides

    Returns:
        Position in storage

    """
    position = 0
    for i, stride in zip(index, strides):
        position += i * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    remaining_ord = int(ordinal)
    for i in range(len(shape) - 1, -1, -1):
        dim_shape = shape[i]
        out_index[i] = remaining_ord % dim_shape
        remaining_ord //= dim_shape


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast

    """
    if len(shape1) < len(shape2):
        smaller = shape1
        larger = shape2
    else:
        smaller = shape2
        larger = shape1

    len_diff = len(larger) - len(smaller)
    smaller = [1] * len_diff + list(smaller)

    res = []
    for s1, s2 in zip(smaller, larger):
        if s1 == s2:
            res.append(s1)
        elif s1 == 1:
            res.append(s2)
        elif s2 == 1:
            res.append(s1)
        else:
            raise IndexingError(f"Cannot broadcast incompatible shapes {s1} and {s2}")

    return tuple(res)


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
        big_index : multidimensional index of bigger tensor
        big_shape : tensor shape of bigger tensor
        shape : tensor shape of smaller tensor
        out_index : multidimensional index of smaller tensor

    Returns:
        None

    """
    for i in range(len(shape)):
        if shape[i] > 1:
            offset = len(big_shape) - len(shape) + i
            out_index[i] = big_index[offset]
        else:
            out_index[i] = 0


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Given a shape of a N-dim tensor, compute its contiguous strides layout."""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))
