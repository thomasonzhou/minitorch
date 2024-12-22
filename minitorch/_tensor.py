"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy import array

from minitorch.core import operators
from minitorch.autograd import Variable, backpropagate
import minitorch
from minitorch._tensor_helpers import IndexingError, Index, to_index
import random
import numba

from ._tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Tanh,
    GeLU,
    Sigmoid,
    Sum,
    View,
)
from ._tensor_helpers import strides_from_shape, index_to_position, shape_broadcast
from ._tensor_functions import History

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=np.float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(operators.prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        return TensorData(
            storage=self._storage,
            shape=tuple(self.shape[i] for i in order),
            strides=tuple(self.strides[i] for i in order),
        )

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    device: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        device: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert device is not None
        self._tensor = v
        self.history = back
        self.device = device
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = device

    def requires_grad_(self, x: bool) -> None:
        self.history = History()

    def requires_grad(self) -> bool:
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns:
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    @property
    def shape(self) -> UserShape:
        """Returns:
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns:
        int : size of the tensor

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns:
        int : dimensionality of the tensor

        """
        return self._tensor.dims

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), device=self.device)
        else:
            b._type_(self.device)
            c = b
        return c

    # Functions
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __rsub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self._ensure_tensor(b), -self)

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __le__(self, b: TensorLike) -> Tensor:
        return self < b or self == b

    def __ge__(self, b: TensorLike) -> Tensor:
        return self > b or self == b

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __hash__(self) -> int:
        return self.unique_id

    def all(self, dim: Optional[int] = None) -> Tensor:
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return ReLU.apply(self)

    def log(self) -> Tensor:
        return Log.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)

    def tanh(self) -> Tensor:
        return Tanh.apply(self)

    def gelu(self) -> Tensor:
        return GeLU.apply(self)

    def item(self) -> float:
        assert self.size == 1
        return self[0]

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Compute the sum over dimension `dim`"""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Compute the mean over dimension `dim`"""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permute tensor dimensions to *order"""
        return Permute.apply(self, minitorch.tensor(list(order), device=self.device))

    def view(self, *shape: int) -> Tensor:
        """Change the shape of the tensor to a new shape with the same size"""
        return View.apply(self, minitorch.tensor(list(shape), device=self.device))

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, device: TensorBackend) -> None:
        self.device = device
        if device.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, device=self.device)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        device: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), device=device)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Parameters
        ----------
            other : backward tensor (must broadcast with self)

        Returns
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.device.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.device.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, device=self.device)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        def zero(shape: UserShape) -> Tensor:
            return Tensor.make([0.0] * int(operators.prod(shape)), shape, device=self.device)

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.device)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        return Tensor(self._tensor, device=self.device)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0] * int(operators.prod(self.shape)), self.shape, device=self.device
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [(inp, inp.expand(self._ensure_tensor(d_in))) for inp, d_in in zip(h.inputs, x)]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), device=self.device)
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:  # pragma: no cover
        """Reset the derivative on this variable."""
        self.grad = None


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = minitorch.zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
