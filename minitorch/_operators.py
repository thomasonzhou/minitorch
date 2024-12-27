"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Sequence


def mul(x: float, y: float) -> float:
    """Compute the product of x and y."""
    return x * y


def id(x: float) -> float:
    """Return the same value."""
    return x


def add(x: float, y: float) -> float:
    """Compute the sum of x and y."""
    return x + y


def neg(x: float) -> float:
    """Compute the negation of x."""
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Determine if x is less than y."""
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    """Determine if x is equal to y."""
    if x == y:
        return 1.0
    else:
        return 0.0


def boolean_not(x: bool) -> bool:
    """Change False to True and True to False"""
    return not x


def max(x: float, y: float) -> float:
    """Find the max of x and y."""
    if x < y:
        return y
    return x


def is_close(x: float, y: float) -> bool:
    """Determine if x is close to y."""
    threshold = 1e-2
    return abs(x - y) < threshold


def exp(x: float) -> float:
    """Compute the exponential of x."""
    return math.exp(x)


def relu(x: float) -> float:
    """Compute the ReLU of x."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Compute the log of x."""
    return math.log(x)


def inv(x: float) -> float:
    """Compute the reciprocal of x."""
    return 1.0 / x


def sigmoid(x: float) -> float:
    """Compute the sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    """Compute the hyperbolic tangent of x"""
    return math.tanh(x)
    # pos_exp = math.exp(x)
    # neg_exp = math.exp(-x)
    # return (pos_exp - neg_exp) / (pos_exp + neg_exp)


def tanh_back(x: float, deriv: float) -> float:
    """Compute the derivative of tanh times a value."""
    return (1.0 - math.tanh(x) ** 2) * deriv


def gelu(x: float) -> float:
    """Approximation of GeLU, a smooth alternative to ReLU
    https://paperswithcode.com/method/gelu"""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))


def gelu_back(x: float, deriv: float) -> float:
    """Compute the derivative of approximated GeLU times a value"""
    a = math.sqrt(2.0 / math.pi)
    b = 0.044715
    z = a * (x + b * (x**3))
    tanh_z = math.tanh(z)
    sech_z = 1 - (tanh_z**2)
    gelu_deriv = 0.5 * (1 + tanh_z) + 0.5 * x * a * (1 + 3 * b * x * x) * sech_z
    return gelu_deriv * deriv


def log_back(x: float, deriv: float) -> float:
    """Compute the derivative of log times a value."""
    return deriv / x


def inv_back(x: float, deriv: float) -> float:
    """Compute the derivative of inv times a value."""
    return -deriv / (x**2)


def relu_back(x: float, deriv: float) -> float:
    """Compute the derivative of ReLU times a value."""
    if x <= 0:
        return 0.0
    return deriv


def map(f: Callable[[float], float], l1: list[float]) -> list[float]:
    """Apply a function f to each element of list l."""
    return [f(val) for val in l1]


def zipWith(f: Callable[[float, float], float], l1: list[float], l2: list[float]) -> list[float]:
    """Apply a function f to combine lists l1 and l2."""
    return [f(val1, val2) for val1, val2 in zip(l1, l2)]


def reduce(f: Callable[[float, float], float], l: Sequence[float]) -> float:
    """Reduce a list l to one value using repeated calls to f."""
    if len(l) == 0:
        return 0

    curr = l[0]
    for idx in range(1, len(l)):
        curr = f(curr, l[idx])
    return curr


def negList(l: list[float]) -> list[float]:
    """Negate each element of list l."""
    return map(neg, l)


def addLists(l1: list[float], l2: list[float]) -> list[float]:
    """Compute element-wise sum of lists l1 and l2."""
    return zipWith(add, l1, l2)


def sum(l: list[float]) -> float:
    """Compute the sum of list l."""
    return reduce(add, l)


def prod(l: Sequence[float]) -> float:
    """Compute the product of list l."""
    return reduce(mul, l)
