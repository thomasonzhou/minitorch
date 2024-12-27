from minitorch._tensor import Tensor


def where(condition: Tensor, input: Tensor, other: Tensor):
    assert condition.shape == input.shape == other.shape
    return condition * input + (~condition) * other
    # res = input.contiguous()
    # for i in range(len(condition)):
    #     if not condition[i]:
    #         res[i] = other[i]
    # return res


def isclose(input: Tensor, other: Tensor, rtol: float = 1e-5, atol: float = 1e-08) -> Tensor:
    return abs(input - other) <= atol + rtol * abs(other)
