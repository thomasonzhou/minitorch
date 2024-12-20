"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable

# ## Task 0.1


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


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


def max(x: float, y: float) -> float:
    """Find the max of x and y."""
    if lt(x, y):
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
    return 1.0/(1.0 + math.exp(-x))


def log_back(x: float, deriv: float) -> float:
    """Compute the derivative of log times a value."""
    return deriv / x


def inv_back(x: float, deriv: float) -> float:
    """Compute the derivative of inv times a value."""
    return - deriv / (x ** 2)


def relu_back(x: float, deriv: float) -> float:
    """Compute the derivative of ReLU times a value."""
    if x <= 0:
        return 0.0
    return deriv


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
def map(f: Callable[float, float], l: list) -> list:
    """Apply a function f to each element of list l."""
    return [f(val) for val in l]


def zipWith(f: Callable[[float, float], float], l1: list, l2: list) -> list:
    """Apply a function f to combine lists l1 and l2."""
    return [f(val1, val2) for val1, val2 in zip(l1, l2)]


def reduce(f: Callable[[float, float], float], l: list) -> float:
    """Reduce a list l to one value using repeated calls to f."""
    match len(l):
        case 0:
            return 0
        case 1:
            return l[0]
        case _:
            curr = f(l[0], l[1])
            for idx in range(2, len(l)):
                curr = f(curr, l[idx])
            return curr


def negList(l: list) -> list:
    """Negate each element of list l."""
    return map(neg, l)


def addLists(l1: list, l2: list) -> list:
    """Compute element-wise sum of lists l1 and l2."""
    return zipWith(add, l1, l2)


def sum(l: list) -> list:
    """Compute the sum of list l."""
    return reduce(add, l)


def prod(l: list) -> list:
    """Compute the product of list l."""
    return reduce(mul, l)
