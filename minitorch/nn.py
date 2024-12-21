from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    return (
        input.contiguous()
        .view(batch, channel, height, new_width, kw)
        .permute(0, 1, 3, 2, 4)  # reorder height to be next to kh when splitting
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute the mean over the last dimension of tiled input"""
    tiled_tensor, new_height, new_width = tile(input, kernel)
    batch, channel, _new_height, _new_width, _kk = tiled_tensor.shape
    return tiled_tensor.mean(4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a 1-hot tensor highlighting the maximum value in a tensor
    Selects the most recent of the maximums if there exist multiple
    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Reduce to find the maxmimum on each dimension"""
        ctx.save_for_backward(t1, dim)
        return max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute argmax and multiply with the gradient"""
        (
            t1,
            dim,
        ) = ctx.saved_tensors
        return grad_output * argmax(t1, int(dim.item())), grad_output.zeros(grad_output.shape)


def max(t: Tensor, dim: int | None = None) -> Tensor:
    """Compute the maximum along a given axis"""
    if dim is None:
        return Max.apply(t.contiguous().view(t.size), t._ensure_tensor(0))
    else:
        return Max.apply(t, t._ensure_tensor(dim))


def softmax(t: Tensor, dim: int) -> Tensor:
    """Continuous weighting of the probability distribution, a.k.a. argsoftmax"""
    return t.exp() / t.exp().sum(dim)


def logsoftmax(t: Tensor, dim: int) -> Tensor:
    """More numerically stable version of softmax"""
    maxval = max(t, dim)
    return t - ((t - maxval).exp().sum(dim).log() + maxval)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute the maximum over the last dimension of tiled input"""
    tiled_tensor, new_height, new_width = tile(input, kernel)

    return max(tiled_tensor, 4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(t: Tensor, probability: float, ignore: bool = False) -> Tensor:
    """Drop values with likelihood 0.0 to 1.0, where 1.0 represents dropping all values"""
    assert 0.0 <= probability <= 1.0
    if ignore or probability == 0:
        return t
    drop = 1.0 * (probability <= rand(t.shape, t.backend))
    return t * drop
