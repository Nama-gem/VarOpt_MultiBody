import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from functions.generate_square_array import ConstructSquareArray
from functions import dressing_functions_spin as df


def dense_xy(arr, echo):
    """Independent Pauli/Kronecker reference in ascending bit-i order."""
    eye = np.eye(2)
    ops = {'x': np.array([[0, 1], [1, 0]], complex),
           'y': np.array([[0, 1j], [-1j, 0]], complex),
           'z': np.diag([-1, 1])}
    h = arr.constant*np.eye(arr.full_dim, dtype=complex)
    for sites, coefficient in arr.pauli_z.items():
        if len(sites) == 2:
            axes = ('x', 'y')
        elif echo and len(sites)%2:
            continue
        else:
            axes = ('z',)
        for axis in axes:
            term = np.array([[1.]])
            for i in reversed(range(arr._L)):
                term = np.kron(term, ops[axis] if i in sites else eye)
            h += coefficient*term
    return h


class XYTests(unittest.TestCase):
    def test_dense_reference_both_boundaries_and_backends(self):
        for boundary, backend, echo in product(('open', 'periodic'), ('numpy', 'quspin'), (False, True)):
            with self.subTest(boundary=boundary, backend=backend, echo=echo):
                arr = ConstructSquareArray(3, 2, 1.3, .7, 4, boundary=boundary, backend=backend)
                full = dense_xy(arr, echo)
                operator = arr.xy_operator(echo=echo)
                self.assertIs(operator, arr.xy_operator(echo=echo))
                projection = arr.to_full(np.eye(arr.dim))
                reduced = operator @ np.eye(arr.dim)
                assert_allclose(reduced, projection.conj().T @ full @ projection, atol=2e-12)
                assert_allclose(reduced, reduced.conj().T, atol=1e-13)
                rng = np.random.default_rng(11)
                psi = rng.normal(size=arr.dim)+1j*rng.normal(size=arr.dim)
                psi /= np.linalg.norm(psi)
                actual = arr.evolve_xy(psi, 7., echo=echo)
                expected = expm(-7j*full) @ arr.to_full(psi)
                assert_allclose(arr.to_full(actual), expected, atol=2e-12)
                assert_allclose(np.linalg.norm(actual), 1., atol=1e-13)
                assert_allclose(arr.evolve_xy(actual, -7., echo=echo)[:, 0], psi, atol=1e-12)

    def test_exchange_normalization_and_echo_definition(self):
        arr = ConstructSquareArray(2, 1, 1.4, .6, 2, backend='numpy')
        coupling = arr.pauli_z[(0, 1)]
        h = arr.xy_operator(echo=True) @ np.eye(arr.dim)
        assert_allclose(h[1, 2], 2*coupling, atol=1e-14)
        assert_allclose(h[0, 3], 0., atol=1e-14)
        assert_allclose(h.diagonal(), arr.constant, atol=1e-14)
        # Nonuniform many-body z terms generally do not commute with exchange.
        arr = ConstructSquareArray(3, 2, 1.3, .7, 4, backend='numpy')
        h = dense_xy(arr, False)
        even = dense_xy(arr, True)
        assert_allclose(even, (h+h[::-1, ::-1])/2, atol=1e-13)
        odd = (h-h[::-1, ::-1])/2
        self.assertGreater(np.linalg.norm(even@odd-odd@even), 1e-8)
        # Exchange conserves magnetization, even with retained diagonal z terms.
        jz = np.diag(arr._jz)
        assert_allclose(jz@h-h@jz, 0., atol=1e-13)

    def test_state_gradient_and_objective_hessian(self):
        for backend in ('numpy', 'quspin'):
            arr = ConstructSquareArray(3, 2, 1.4, .65, 4, backend=backend)
            gates = ['Ry', 'XY', 'Rx', 'XY_echo', 'Iz', 'Rz']
            theta = np.array([.3, 8., .5, 12., 4., .2])
            f, g, h, psi, dpsi = arr.evaluate_gate_sequence_squeezing(
                theta, gates, hessian=True, return_state=True)
            for i in range(len(theta)):
                step = np.eye(len(theta))[i]*1e-4
                fp, gp = arr.evaluate(theta+step, gates, True)
                fm, gm = arr.evaluate(theta-step, gates, True)
                pp = arr.run_sequence(theta+step, gates)
                pm = arr.run_sequence(theta-step, gates)
                assert_allclose(g[i], (fp-fm)/2e-4, atol=3e-8)
                assert_allclose(h[:, i], (gp-gm)/2e-4, atol=5e-8)
                assert_allclose(dpsi[i], (pp-pm)/2e-4, atol=3e-8)
            assert_allclose(np.linalg.norm(psi), 1., atol=1e-12)
            assert_allclose(h, h.T, atol=1e-13)

    def test_zero_pairs_and_helpers(self):
        for backend in ('numpy', 'quspin'):
            for n, omega in ((1, .6), (2, 0.)):
                arr = ConstructSquareArray(2, 1, 1.4, omega, n, backend=backend)
                psi = arr._initial_state
                assert_allclose(df.apply_H_XY(arr, psi, 5), arr.evolve(psi, 5), atol=1e-13)
                assert_allclose(df.apply_H_XY_echo(arr, psi, 5), arr.evolve(psi, 5, echo=True), atol=1e-13)
            with self.assertRaises(ValueError):
                arr.evolve_xy(psi, np.inf)

    def test_xy_optimizer_both_methods(self):
        arr = ConstructSquareArray(3, 2, 1.4, .7, 3)
        gates = ['XY_echo']
        for hessian in (False, True):
            result = arr.optimize(gates, [10.], hessian=hessian, bounds=[(1, 100)],
                                  options={'maxiter': 80, 'gtol': 1e-8})
            self.assertTrue(result.success, result.message)
            self.assertLess(result.fun, arr.evaluate([10.], gates))
            self.assertTrue(1 <= result.x[0] <= 100)
            if hessian:
                self.assertGreater(result.nhev, 0)


if __name__ == '__main__':
    unittest.main()
