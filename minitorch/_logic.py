from minitorch._tensor import Tensor


def where(condition: Tensor, input: Tensor, other: Tensor) -> Tensor:
    assert condition.shape == input.shape == other.shape
    return condition * input + (~condition) * other

def outer(input: Tensor, other: Tensor) -> Tensor:
    return input[:, None] * other

def isclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-08) -> Tensor:
    return abs(input - other) <= atol + rtol * abs(other)
