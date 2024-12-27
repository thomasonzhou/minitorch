from .module import Module
from minitorch.nn.utils import RParam
from minitorch.backends import with_current_backend


class Linear(Module):
    @with_current_backend
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.weights = RParam(in_features, out_features, device=device)
        self.apply_bias = bias
        if bias:
            self.bias = RParam(out_features, device=device)
        self.out_size = out_features

    def forward(self, x):
        batch, in_size = x.shape
        x = (x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)).view(
            batch, self.out_size
        )
        if self.apply_bias:
            x = x + self.bias.value
        return x
