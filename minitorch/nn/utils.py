

def RParam(*shape, device):
    """Randomly initialize weights of a given shape"""
    r = 0.1 * (minitorch.rand(shape, backend=device) - 0.5)
    return minitorch.Parameter(r)
