from .module import Module
from minitorch.nn.functional import dropout


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return dropout(x, self.p, ignore=not self.training)
