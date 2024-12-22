from minitorch.nn.fast_conv import conv1d, conv2d
from minitorch.nn.utils import RParam
from .module import Module
from minitorch.backends import with_current_backend


class Conv1d(Module):
    @with_current_backend
    def __init__(self, in_channels, out_channels, kernel_width, device):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width, device=device)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        return conv1d(input, self.weights.value) + self.bias.value


class Conv2d(Module):
    @with_current_backend
    def __init__(self, in_channels, out_channels, kh, kw, device):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw, device=device)
        self.bias = RParam(out_channels, 1, 1, device=device)

    def forward(self, input):
        return conv2d(input, self.weights.value) + self.bias.value
