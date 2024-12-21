from dataclasses import dataclass
from typing import Any, Iterable, Tuple
from typing_extensions import Protocol
from collections import deque
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    left_args = vals[:arg]
    right_args = vals[arg + 1 :]
    left_f = f(*left_args, vals[arg] - epsilon, *right_args)
    right_f = f(*left_args, vals[arg] + epsilon, *right_args)
    difference = (right_f - left_f) / (2.0 * epsilon)
    return difference


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    """
    # Kahn's algorithm
    indegrees: dict[Variable, int] = {}
    adj: dict[Variable, list[Variable]] = {variable: []}

    q = deque([variable])
    while len(q) > 0:
        v = q.popleft()
        for i in v.history.inputs:  # type: ignore[attr-defined]
            if i.is_constant():
                continue
            indegrees[i] = indegrees.get(i, 0) + 1
            adj.setdefault(v, []).append(i)
            q.append(i)

    assert len(q) == 0
    q.append(variable)

    res = []
    while len(q) > 0:
        v = q.popleft()
        res.append(v)
        for i in adj.get(v, []):
            indegrees[i] -= 1
            if indegrees[i] == 0:
                q.append(i)

    return res

    # DFS
    # visited = set()
    # res = []

    # dfs approach
    # def dfs(node) -> None:
    #     if node.is_constant() or node in visited:
    #         return
    #     for i in node.history.inputs:
    #         dfs(i)
    #     visited.add(node)
    #     res.append(node)

    # dfs(variable)

    # return reversed(res)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    variable_to_deriv = {variable: deriv}

    for v in topological_sort(variable):
        if v.is_leaf():
            v.accumulate_derivative(variable_to_deriv[v])
        else:
            for in_var, new_deriv in v.chain_rule(variable_to_deriv[v]):
                variable_to_deriv[in_var] = variable_to_deriv.get(in_var, 0) + new_deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return saved values for backward pass"""
        return self.saved_values
