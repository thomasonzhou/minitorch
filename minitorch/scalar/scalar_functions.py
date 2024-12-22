from __future__ import annotations

from typing import TYPE_CHECKING

from minitorch.core import operators
from minitorch.autograd import Context

if TYPE_CHECKING:
    from typing import Tuple, Any
    from .scalar import Scalar, ScalarLike, ScalarHistory


def wrap_tuple(x: Any) -> Tuple[Any]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Tuple[Any]) -> Any:
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars: list[Scalar] = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(Scalar(v, back=None))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = ScalarHistory(cls, ctx, scalars)
        return Scalar(c, back)


class Add(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.inv_back(unwrap_tuple(ctx.saved_values), d_output)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a = unwrap_tuple(ctx.saved_values)
        return d_output * operators.sigmoid(a) * (1.0 - operators.sigmoid(a))


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.relu_back(unwrap_tuple(ctx.saved_values), d_output)


class Tanh(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.tanh(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.tanh_back(unwrap_tuple(ctx.saved_values), d_output)


class GeLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.gelu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.gelu_back(unwrap_tuple(ctx.saved_values), d_output)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return operators.exp(unwrap_tuple(ctx.saved_values))


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return (0.0, 0.0)


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return (0.0, 0.0)
