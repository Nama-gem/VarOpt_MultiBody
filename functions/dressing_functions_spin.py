"""Function-style interface matching the soft-core variational project."""
from .generate_square_array import ConstructSquareArray, VarOptMultiBodyIsing


def evaluate_gate_sequence_squeezing(theta, gate_sequence, arr, gradient=False,
                                    *, return_state=False, hessian=False):
    """Return xi or (xi, dxi); return_state also appends psi and, if requested, dpsi.

    With both flags True the return is (xi, dxi, psi, dpsi).
    dpsi[k] is the analytic final-state derivative with respect to theta[k].
    hessian=True returns (xi, dxi, Hxi), appending (psi, dpsi) if return_state=True.
    """
    return arr.evaluate_gate_sequence_squeezing(
        theta, gate_sequence, gradient, return_state=return_state, hessian=hessian)


def evaluate_gate_sequence(theta, gate_sequence, arr, gradient=False):
    """Return psi or (psi, dpsi), where dpsi has shape (n_gates, arr.dim)."""
    return arr.evaluate_gate_sequence(theta, gate_sequence, gradient=gradient)


def get_squeezing(arr, state, state_grad=None):
    return arr.squeezing(state, state_grad)


def apply_H_Ising(arr, state, t_dress):
    return arr.evolve(state, t_dress)


def apply_H_Ising_echo(arr, state, t_dress):
    return arr.evolve(state, t_dress, echo=True)


def apply_H_Ix(arr, state, t_dress):
    return arr.evolve_x(state, t_dress)


def apply_H_Ix_echo(arr, state, t_dress):
    return arr.evolve_x(state, t_dress, echo=True)


def apply_H_XY(arr, state, t_dress):
    return arr.evolve_xy(state, t_dress)


def apply_H_XY_echo(arr, state, t_dress):
    return arr.evolve_xy(state, t_dress, echo=True)
