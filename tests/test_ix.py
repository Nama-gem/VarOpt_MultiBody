import unittest
from itertools import product
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from functions.generate_square_array import ConstructSquareArray
from functions import dressing_functions_spin as df


def dense_ising_x(arr, echo):
    eye = np.eye(2)
    x = np.array([[0., 1.], [1., 0.]])
    h = arr.constant*np.eye(arr.full_dim)
    for sites, coefficient in arr.pauli_z.items():
        if echo and len(sites)%2:
            continue
        term = np.array([[1.]])
        for site in reversed(range(arr._L)):
            term = np.kron(term, x if site in sites else eye)
        h += coefficient*term
    return h


class IxTests(unittest.TestCase):
    def test_dense_reference_all_orders_and_boundaries(self):
        for backend, boundary, echo in product(('numpy', 'quspin'), ('open', 'periodic'), (False, True)):
            with self.subTest(backend=backend, boundary=boundary, echo=echo):
                arr = ConstructSquareArray(3, 2, 1.3, .65, 4, backend=backend, boundary=boundary)
                h = dense_ising_x(arr, echo)
                rng = np.random.default_rng(23)
                state = rng.normal(size=(arr.dim, 2))+1j*rng.normal(size=(arr.dim, 2))
                state /= np.linalg.norm(state, axis=0)
                assert_allclose(arr.to_full(arr.apply_ising_x(state, echo=echo)),
                                h @ arr.to_full(state), atol=1e-12)
                evolved = arr.evolve_x(state, 8., echo=echo)
                assert_allclose(arr.to_full(evolved), expm(-8j*h) @ arr.to_full(state), atol=2e-12)
                assert_allclose(arr.evolve_x(evolved, -8., echo=echo), state, atol=2e-12)
                gate = 'Ix_echo' if echo else 'Ix'
                assert_allclose(arr.run_sequence([8.], [gate], state=state[:, 0]), evolved[:, 0], atol=1e-12)

    def test_echo_and_default_initial_state(self):
        for backend in ('numpy', 'quspin'):
            arr = ConstructSquareArray(3, 2, 1.3, .65, 4, backend=backend)
            rng = np.random.default_rng(5)
            state = rng.normal(size=arr.dim)+1j*rng.normal(size=arr.dim)
            state /= np.linalg.norm(state)
            # A midpoint z pi pulse flips every x; a final -pi restores the frame.
            physical = arr.evolve_x(state, 3.)
            physical = arr.rotate(physical, 'z', np.pi)
            physical = arr.evolve_x(physical, 3.)
            physical = arr.rotate(physical, 'z', -np.pi)
            assert_allclose(physical, arr.evolve_x(state, 6., echo=True), atol=2e-12)
            for echo in (False, True):
                energy = arr.constant+sum(v for sites, v in arr.pauli_z.items()
                                         if not echo or len(sites)%2 == 0)
                expected = np.exp(-6j*energy)*arr._initial_state
                helper = df.apply_H_Ix_echo if echo else df.apply_H_Ix
                assert_allclose(helper(arr, arr._initial_state, 6.), expected, atol=2e-12)
            with self.assertRaises(ValueError):
                arr.evolve_x(state, np.nan)

    def test_state_gradients_and_hessian(self):
        for backend in ('numpy', 'quspin'):
            arr = ConstructSquareArray(3, 2, 1.3, .65, 4, backend=backend)
            gates = ['Ry', 'Ix', 'Iz_echo', 'Rx', 'Ix_echo', 'Rz']
            theta = np.array([.35, 8., 10., .5, 12., -.2])
            f, g, h, psi, dpsi = arr.evaluate_gate_sequence_squeezing(
                theta, gates, hessian=True, return_state=True)
            for i in range(len(theta)):
                step = np.eye(len(theta))[i]*1e-4
                fp, gp = arr.evaluate(theta+step, gates, True)
                fm, gm = arr.evaluate(theta-step, gates, True)
                pp = arr.run_sequence(theta+step, gates)
                pm = arr.run_sequence(theta-step, gates)
                assert_allclose(dpsi[i], (pp-pm)/2e-4, atol=4e-8)
                assert_allclose(g[i], (fp-fm)/2e-4, atol=4e-8)
                assert_allclose(h[:, i], (gp-gm)/2e-4, atol=8e-8)
            assert_allclose(np.linalg.norm(psi), 1., atol=1e-12)
            assert_allclose(h, h.T, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
