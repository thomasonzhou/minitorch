from minitorch._tensor import Tensor


def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
    """Given a tensor of booleans, return a tensor with value of input if True and other if False"""
    assert condition.shape == input.shape == other.shape
    return condition * input + (~condition) * other

def outer(input: Tensor, vec2: Tensor) -> Tensor:
    """Outer product of input and vec2, with no broadcast support"""
    return input[:, None] * vec2

def isclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-08) -> Tensor:
    """Check if an absolute difference is within a threshold"""
    return abs(input - other) <= atol + rtol * abs(other)
