"""Analytic second-order differentiation of the scalar squeezing objective.

State derivatives are supplied by gate propagation. Only small scalar
value/gradient/Hessian objects are used here; no autodiff dependency or
numerical differentiation is needed.
"""
import numpy as np


class _Jet:
    """Real scalar and its first and second parameter derivatives."""
    def __init__(self, value, gradient, hessian):
        self.value = float(value)
        self.gradient = np.asarray(gradient, float)
        self.hessian = np.asarray(hessian, float)

    def __add__(self, other):
        if not isinstance(other, _Jet):
            return _Jet(self.value+other, self.gradient, self.hessian)
        return _Jet(self.value+other.value, self.gradient+other.gradient,
                    self.hessian+other.hessian)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, _Jet):
            return _Jet(self.value*other, self.gradient*other, self.hessian*other)
        cross = np.outer(self.gradient, other.gradient)
        return _Jet(self.value*other.value,
                    self.gradient*other.value + self.value*other.gradient,
                    self.hessian*other.value + self.value*other.hessian + cross + cross.T)

    __rmul__ = __mul__

    def __pow__(self, power):
        # Powers below are 2, -0.5 and 0.5; fractional powers need positive values.
        if self.value <= 0 and power != 2:
            raise ValueError('Squeezing Hessian requires positive mean-spin length and variance.')
        first = power*self.value**(power-1)
        second = power*(power-1)*self.value**(power-2)
        return _Jet(self.value**power, first*self.gradient,
                    first*self.hessian + second*np.outer(self.gradient, self.gradient))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def squeezing_second_order(arr, psi, first, second, pairs):
    """Return (xi, gradient, Hessian) from column-wise state derivatives.

    second[:,q] is d2psi/dtheta_i dtheta_j for (i,j)=pairs[q].
    All spin moments, centering, mean-spin direction, and the minimizing
    transverse eigenvector are differentiated. At a degenerate transverse
    eigenvalue a classical Hessian is generally undefined; raise explicitly.
    """
    p = first.shape[1]
    rows = np.array([i for i, j in pairs], dtype=int)
    cols = np.array([j for i, j in pairs], dtype=int)

    def moment(operator):
        op_psi = operator(psi)[:, 0]
        value = np.vdot(psi, op_psi).real
        gradient = 2*np.real(op_psi.conj() @ first)
        packed = 2*np.real(op_psi.conj() @ second)
        hessian = np.empty((p, p))
        hessian[rows, cols] = packed
        hessian[cols, rows] = packed
        hessian += 2*np.real(first.conj().T @ operator(first))
        return _Jet(value, gradient, (hessian+hessian.T)/2)

    mean = [moment(lambda state, axis=axis: arr.collective(axis, state)) for axis in 'xyz']
    length2 = _dot(mean, mean)
    if length2.value < 1e-24:
        raise ValueError('Squeezing Hessian is undefined at vanishing mean spin.')
    normal = [v*length2**(-0.5) for v in mean]
    covariance = [[None]*3 for _ in range(3)]
    for i, axis in enumerate('xyz'):
        for j in range(i, 3):
            axis2 = 'xyz'[j]
            def symmetrized(state, a=axis, b=axis2):
                if a == b:
                    return arr.collective(a, arr.collective(a, state))
                return (arr.collective(a, arr.collective(b, state)) +
                        arr.collective(b, arr.collective(a, state)))/2
            covariance[i][j] = covariance[j][i] = moment(symmetrized) - mean[i]*mean[j]

    # A locally smooth orthonormal transverse frame. Its reference axis is
    # held fixed for differentiation; the final eigenvalue is frame invariant.
    reference = np.eye(3)[np.argmin([abs(v.value) for v in normal])]
    u = _cross(normal, reference)
    inverse_norm = _dot(u, u)**(-0.5)
    u = [v*inverse_norm for v in u]
    v = _cross(normal, u)
    def quadratic(left, right):
        return sum(left[i]*covariance[i][j]*right[j] for i in range(3) for j in range(3))
    aa, bb, ab = quadratic(u, u), quadratic(v, v), quadratic(u, v)
    gap2 = (aa-bb)**2 + 4*ab**2
    scale = max(1., abs(aa.value), abs(bb.value))
    if gap2.value <= (1e-10*scale)**2:
        raise ValueError('Squeezing Hessian is undefined or ill-conditioned at degenerate '
                         'transverse variances. Start with a nonzero interacting sequence.')
    variance = (aa+bb-gap2**0.5)*0.5
    if variance.value <= 0:
        raise ValueError('Squeezing Hessian requires strictly positive transverse variance.')
    xi = (arr._L*variance)**0.5 * length2**(-0.5)
    return xi.value, xi.gradient, (xi.hessian+xi.hessian.T)/2
