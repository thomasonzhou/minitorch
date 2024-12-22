from .module import Module
from minitorch.nn.functional import softmax


class SoftMax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return softmax(x, self.dim)


def ReLU(Module):
    def forward(self, x):
        return x.relu()


def Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()
