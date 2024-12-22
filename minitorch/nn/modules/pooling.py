from .module import Module
from minitorch.nn.functional import maxpool2d, avgpool2d


class _MaxPoolNd(Module):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size


class MaxPool2d(_MaxPoolNd):
    def forward(self, x):
        return maxpool2d(x, self.kernel_size)


class AvgPool2d(_MaxPoolNd):
    def forward(self, x):
        return avgpool2d(x, self.kernel_size)
