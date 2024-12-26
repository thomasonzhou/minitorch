import minitorch.autograd
import minitorch.backends
import minitorch.core
import minitorch.nn
import minitorch.optim
import minitorch.scalar
import minitorch.utils  # noqa: F401

from minitorch.backends import with_current_backend
from minitorch._tensor_to_wrap import zeros, rand, tensor, arange

zeros = with_current_backend(zeros)
rand = with_current_backend(rand)
tensor = with_current_backend(tensor)
arange = with_current_backend(arange)
