import minitorch.autograd
import minitorch.backends
import minitorch.nn
import minitorch.optim
import minitorch.scalar
import minitorch.utils  # noqa: F401

from minitorch.backends import with_current_backend
from minitorch._tensor_to_wrap import zeros, ones, rand, tensor, arange

newaxis = None

zeros = with_current_backend(zeros)
ones = with_current_backend(ones)
rand = with_current_backend(rand)
tensor = with_current_backend(tensor)
arange = with_current_backend(arange)

from minitorch._logic import where, outer, isclose  # noqa: F401
