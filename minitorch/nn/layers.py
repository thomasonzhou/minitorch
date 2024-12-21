import minitorch
from minitorch.config import DEFAULT_BACKEND
from .fast_conv import conv2d

def RParam(*shape, device):
    """Randomly initialize weights of a given shape"""
    r = 0.1 * (minitorch.rand(shape, backend=device) - 0.5)
    return minitorch.Parameter(r)

class Linear(minitorch.Module):
    def __init__(self, in_features, out_features, bias=True, device=DEFAULT_BACKEND, dtype=None):
        super().__init__()
        self.weights = RParam(in_features, out_features, device=device)
        self.apply_bias = bias
        if bias:
            self.bias = RParam(out_features, device=device)
        self.out_size = out_features

    def forward(self, x):
        batch, in_size = x.shape
        x =(x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size)
        if self.apply_bias:
            x = x + self.bias.value
        return x


class Conv2d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kh, kw, device=DEFAULT_BACKEND):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kh, kw, device=device)
        self.bias = RParam(out_channels, 1, 1, device=device)

    def forward(self, input):
        return conv2d(input, self.weights.value) + self.bias.value
