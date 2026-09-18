import unittest
from itertools import combinations
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm
from functions.generate_square_array import ConstructSquareArray
from functions import dressing_functions_spin as df


class TestMultiBody(unittest.TestCase):
    def setUp(self):
        self.arr = ConstructSquareArray(2, 2, 1.3, 0.65, 4, backend='numpy')

    def test_exact_cluster_spectrum_and_open_geometry(self):
        a = self.arr
        for mask in range(a.dim):
            sites = [i for i in range(a._L) if mask >> i & 1]
            m = len(sites)
            h = np.zeros((1 << m, 1 << m))
            for b in range(1 << m):
                h[b, b] = b.bit_count()
                for i, j in combinations(range(m), 2):
                    si, sj = sites[i], sites[j]
                    r2 = (si%2-sj%2)**2 + (si//2-sj//2)**2
                    h[b, b] += 2*a.Rb**6/r2**3*((b >> i)&1)*((b >> j)&1)
                for i in range(m):
                    h[b, b ^ (1 << i)] = a.Omega/2
            assert_allclose(a.h_ising[mask], np.linalg.eigvalsh(h)[0], atol=2e-12)
        chain = ConstructSquareArray(4, 1, 1.2, 0.5, 2, backend='numpy')
        self.assertGreater(abs(chain.occupation[(0, 1)]), abs(chain.occupation[(0, 3)]))

    def test_pauli_expansion_and_echo(self):
        a = self.arr
        ids = np.arange(a.dim)
        expected, even = np.full(a.dim, a.constant), np.full(a.dim, a.constant)
        for sites, v in a.pauli_z.items():
            term = v*np.prod([2*((ids >> i)&1)-1 for i in sites], axis=0)
            expected += term
            if len(sites)%2 == 0:
                even += term
        assert_allclose(a.h_ising, expected, atol=1e-13)
        assert_allclose(a.h_echo, even, atol=1e-13)
        # Odd clusters must contribute to even Pauli terms.
        a2 = ConstructSquareArray(2, 2, 1.3, .65, 2, backend='numpy')
        a3 = ConstructSquareArray(2, 2, 1.3, .65, 3, backend='numpy')
        self.assertGreater(abs(a2.pauli_z[(0, 1)]-a3.pauli_z[(0, 1)]), 1e-7)
        rng = np.random.default_rng(12)
        psi = rng.normal(size=a.dim)+1j*rng.normal(size=a.dim)
        psi /= np.linalg.norm(psi)
        t = 2.4
        physical = a.evolve(a.rotate(a.evolve(psi, t/2), 'x', np.pi), t/2)
        restored = a.rotate(physical, 'x', -np.pi)
        assert_allclose(restored, a.evolve(psi, t, echo=True), atol=1e-13)
        assert_allclose(physical[:, 0], a.run_sequence([t], ['Dz'], state=psi), atol=1e-13)

    def test_rotations_dense_and_commutators(self):
        a = self.arr
        identity = np.eye(a.dim)
        ops = [a.collective(axis, identity) for axis in 'xyz']
        assert_allclose(ops[0]@ops[1]-ops[1]@ops[0], 1j*ops[2], atol=1e-14)
        for axis, j in zip('xyz', ops):
            assert_allclose(a.rotate(identity, axis, .7), expm(-.7j*j), atol=1e-14)

    def test_state_and_squeezing_gradients(self):
        a = self.arr
        gates = ['Ry', 'Iz', 'Rx', 'Iz_echo', 'Rz', 'Dz']
        theta = np.array([.3, 4.1, .73, 9.4, -.31, 2.6])
        xi, dxi, psi, grad = df.evaluate_gate_sequence_squeezing(
            theta, gates, a, gradient=True, return_state=True)
        self.assertTrue(np.isfinite(xi))
        for i in range(len(theta)):
            step = np.eye(len(theta))[i]*1e-5
            plus = a.evaluate_gate_sequence(theta+step, gates)
            minus = df.evaluate_gate_sequence(theta-step, gates, a)
            assert_allclose(grad[i], (plus-minus)/2e-5, atol=2e-10)
            assert_allclose(dxi[i], (a.squeezing(plus)-a.squeezing(minus))/2e-5, atol=2e-8)
        assert_allclose(np.linalg.norm(psi), 1, atol=1e-13)
        rotated = a.rotate(a.rotate(psi, 'y', .89), 'z', .27)
        assert_allclose(a.squeezing(rotated), xi, atol=1e-12)

    def test_limits_validation_and_optimizer(self):
        a = self.arr
        self.assertAlmostEqual(a.squeezing(a._initial_state), 1)
        psi, grad = a.run_sequence([], [], gradient=True)
        self.assertEqual(grad.shape, (0, a.dim))
        for rb, omega in ((0., .5), (1., 0.)):
            model = ConstructSquareArray(2, 1, rb, omega, 2, backend='numpy')
            assert_allclose(model.h_echo, model.constant, atol=1e-13)
        with self.assertRaises(ValueError):
            ConstructSquareArray(5, 5, 1, .3, 3)
        with self.assertRaises(ValueError):
            a.run_sequence([1], ['invalid'])
        with self.assertRaises(ValueError):
            a.run_sequence([1, 2], ['Iz'])
        initial = a.evaluate([10.], ['Iz_echo'])
        result = a.optimize(['Iz_echo'], [10.], bounds=[(1., 150.)], options={'maxiter': 30})
        self.assertLess(result.fun, initial)


if __name__ == '__main__':
    unittest.main()
