import unittest
from itertools import combinations
import numpy as np
from numpy.testing import assert_allclose
from functions.generate_square_array import ConstructSquareArray
from functions import dressing_functions_spin as df


class HessianAndBoundaryTests(unittest.TestCase):
    def test_hessian_and_mixed_state_derivatives(self):
        for backend in ('numpy', 'quspin'):
            a = ConstructSquareArray(2, 2, 1.3, .65, 4, backend=backend)
            gates = ['Ry', 'Iz', 'Rx', 'Iz_echo', 'Rz', 'Dz']
            theta = np.array([.3, 4.1, .73, 9.4, -.31, 2.6])
            f, g, h, psi, dpsi = df.evaluate_gate_sequence_squeezing(
                theta, gates, a, hessian=True, return_state=True)
            f0, g0 = a.evaluate(theta, gates, True)
            assert_allclose(f, f0, atol=1e-12)
            assert_allclose(g, g0, atol=1e-12)
            assert_allclose(h, h.T, atol=1e-13)
            _, first, second, pairs = a._run_sequence_second_order(theta, gates)
            assert_allclose(first.T, dpsi, atol=1e-13)
            for j in range(len(theta)):
                step = np.eye(len(theta))[j]*1e-4
                gp = a.evaluate(theta+step, gates, True)[1]
                gm = a.evaluate(theta-step, gates, True)[1]
                assert_allclose(h[:, j], (gp-gm)/2e-4, atol=3e-8)
                dp = a.run_sequence(theta+step, gates, gradient=True)[1]
                dm = a.run_sequence(theta-step, gates, gradient=True)[1]
                for i in range(j+1):
                    assert_allclose(second[:, pairs.index((i, j))], (dp[i]-dm[i])/2e-4,
                                    atol=3e-8)
            assert_allclose(np.linalg.norm(psi), 1., atol=1e-13)
            with self.assertRaisesRegex(ValueError, 'degenerate'):
                a.evaluate_gate_sequence_squeezing([0.], ['Iz_echo'], hessian=True)

    def test_reduced_basis_matches_full_for_both_boundaries(self):
        for boundary in ('open', 'periodic'):
            full = ConstructSquareArray(4, 2, 1.4, .6, 3, boundary=boundary, backend='numpy')
            red = ConstructSquareArray(4, 2, 1.4, .6, 3, boundary=boundary)
            self.assertLess(red.dim, full.dim)
            rng = np.random.default_rng(17)
            state = rng.normal(size=red.dim)+1j*rng.normal(size=red.dim)
            state /= np.linalg.norm(state)
            full_state = red.to_full(state)
            assert_allclose(red.from_full(full_state)[:, 0], state, atol=1e-13)
            assert_allclose(red.to_full(red._initial_state), full._initial_state, atol=1e-13)
            for axis in 'xyz':
                assert_allclose(red.to_full(red.collective(axis, state)),
                                full.collective(axis, full_state), atol=1e-12)
            gates = ['Ry', 'Iz', 'Rx', 'Iz_echo', 'Rz', 'Dz']
            theta = [.3, 4.1, .73, 9.4, -.31, 2.6]
            psi, grad = red.run_sequence(theta, gates, state=state, gradient=True)
            psi_full, grad_full = full.run_sequence(theta, gates, state=full_state, gradient=True)
            assert_allclose(red.to_full(psi)[:, 0], psi_full, atol=1e-12)
            assert_allclose(red.to_full(grad.T).T, grad_full, atol=1e-12)
            fr, gr, hr = red.evaluate_gate_sequence_squeezing(theta, gates, hessian=True)
            ff, gf, hf = full.evaluate_gate_sequence_squeezing(theta, gates, hessian=True)
            assert_allclose(fr, ff, atol=1e-11)
            assert_allclose(gr, gf, atol=1e-11)
            assert_allclose(hr, hf, atol=1e-10)
            nonsymmetric = np.zeros(full.dim)
            nonsymmetric[1] = 1
            with self.assertRaisesRegex(ValueError, 'outside'):
                red.from_full(nonsymmetric)

    def test_unreduced_quspin_bit_order(self):
        a = ConstructSquareArray(2, 2, 1.3, .6, 3, use_reflections=False)
        b = ConstructSquareArray(2, 2, 1.3, .6, 3, backend='numpy')
        identity = np.eye(b.dim)
        native = a.from_full(identity)
        assert_allclose(a.to_full(native), identity)
        for axis in 'xyz':
            assert_allclose(a.to_full(a.rotate(native, axis, .4)), b.rotate(identity, axis, .4), atol=1e-13)
        assert_allclose(a.to_full(a.evolve(native, 5)), b.evolve(identity, 5), atol=1e-13)

    def test_periodic_clusters_against_independent_spectrum(self):
        # Rectangular torus tests the geometry cache must not interchange axes.
        a = ConstructSquareArray(3, 2, 1.2, .55, 6, boundary='periodic', backend='numpy')
        for mask in range(a.dim):
            sites = [i for i in range(a._L) if mask >> i & 1]
            m = len(sites)
            h = np.zeros((1 << m, 1 << m))
            for b in range(1 << m):
                h[b, b] = b.bit_count()
                for i, j in combinations(range(m), 2):
                    si, sj = sites[i], sites[j]
                    dx, dy = abs(si%3-sj%3), abs(si//3-sj//3)
                    r2 = min(dx, 3-dx)**2+min(dy, 2-dy)**2
                    h[b, b] += 2*a.Rb**6/r2**3*((b >> i)&1)*((b >> j)&1)
                for i in range(m):
                    h[b, b ^ (1 << i)] = a.Omega/2
            assert_allclose(a.h_ising[mask], np.linalg.eigvalsh(h)[0], atol=3e-12)
        periodic = ConstructSquareArray(4, 1, 1.2, .55, 3, boundary='periodic', backend='numpy')
        open_array = ConstructSquareArray(4, 1, 1.2, .55, 3, backend='numpy')
        assert_allclose(periodic.occupation[(0, 3)], periodic.occupation[(0, 1)], atol=1e-13)
        self.assertGreater(abs(periodic.occupation[(0, 3)]), abs(open_array.occupation[(0, 3)]))
        # Periodic triples related by a one-site translation.
        assert_allclose(periodic.occupation[(0, 1, 2)], periodic.occupation[(0, 1, 3)], atol=1e-13)

    def test_bounded_hessian_optimizer(self):
        a = ConstructSquareArray(2, 2, 1.3, .65, 3)
        theta = [10.]
        f0 = a.evaluate(theta, ['Iz_echo'])
        result = a.optimize(['Iz_echo'], theta, hessian=True, bounds=[(1., 150.)],
                            options={'maxiter': 100, 'gtol': 1e-9})
        self.assertTrue(result.success, result.message)
        self.assertGreater(result.nhev, 0)
        self.assertLess(result.fun, f0)
        self.assertTrue(1 <= result.x[0] <= 150)


if __name__ == '__main__':
    unittest.main()
