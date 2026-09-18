import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import Bounds, LinearConstraint
from functions.generate_square_array import ConstructSquareArray


class OptimizerOptionsTests(unittest.TestCase):
    def setUp(self):
        self.arr = ConstructSquareArray(2, 2, 1.3, .65, 3, backend='numpy')
        self.gates = ['Iz_echo']

    def test_bounded_methods(self):
        start = self.arr.evaluate([10.], self.gates)
        for method in ('L-BFGS-B', 'SLSQP', 'Powell', 'Nelder-Mead', 'TNC', 'COBYLA', 'COBYQA'):
            with self.subTest(method=method):
                options = {'maxfun': 200} if method == 'TNC' else {'maxiter': 200}
                result = self.arr.optimize(self.gates, [10.], method=method,
                    bounds=Bounds([1.], [150.]), options=options, tol=1e-7)
                self.assertTrue(np.isfinite(result.fun))
                self.assertLess(result.fun, start)
                self.assertTrue(1.-1e-7 <= result.x[0] <= 150.+1e-7)

    def test_unconstrained_methods(self):
        optimum = self.arr.optimize(self.gates, [10.], bounds=[(1, 150)]).x
        start = optimum*.95
        for method in ('BFGS', 'CG', 'Newton-CG', 'trust-exact', 'trust-ncg', 'trust-krylov', 'dogleg'):
            with self.subTest(method=method):
                result = self.arr.optimize(self.gates, start, method=method,
                    bounds=False, tol=1e-8, options={'maxiter': 50})
                self.assertLessEqual(result.fun, self.arr.evaluate(start, self.gates)+1e-9)
                if method in ('trust-exact', 'trust-ncg', 'trust-krylov', 'dogleg'):
                    self.assertGreater(result.nhev, 0)

    def test_constraint_and_callback(self):
        seen = []
        constraint = LinearConstraint([[1.]], [30.], [30.])
        result = self.arr.optimize(self.gates, [10.], constraints=constraint,
                                  callback=lambda x: seen.append(x.copy()), tol=1e-9)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.x[0], 30., places=7)
        self.assertGreater(len(seen), 0)
        seen.clear()
        def callback(intermediate_result):
            seen.append(intermediate_result.fun)
        result = self.arr.optimize(self.gates, [10.], method='trust-constr', hessian=True,
                    bounds=[(1,150)], callback=callback, options={'maxiter': 100})
        self.assertGreater(result.nhev, 0)
        self.assertGreater(len(seen), 0)

    def test_finite_difference_and_derivative_free_skip_state_gradients(self):
        original = self.arr.evaluate_gate_sequence_squeezing
        for method, gradient in [('Powell', None), ('BFGS', False)]:
            with patch.object(self.arr, 'evaluate_gate_sequence_squeezing', wraps=original) as evaluator:
                self.arr.optimize(self.gates, [10.], method=method, gradient=gradient,
                                  bounds=False, options={'maxiter': 3})
                self.assertGreater(evaluator.call_count, 0)
                for call in evaluator.call_args_list:
                    self.assertFalse(call.kwargs['gradient'])
                    self.assertFalse(call.kwargs['hessian'])

    def test_incompatible_requests(self):
        for kwargs in [dict(method='BFGS'), dict(method='Powell', gradient=True),
                       dict(method='BFGS', hessian=True, bounds=False),
                       dict(method='trust-exact', hessian=False, bounds=False),
                       dict(method='Newton-CG', gradient=False, bounds=False),
                       dict(method='L-BFGS-B', constraints={'type':'ineq', 'fun':lambda x:x[0]}),
                       dict(method='trust-constr', hessian=True, gradient=False),
                       dict(method='unknown')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.arr.optimize(self.gates, [10.], **kwargs)
        with self.assertRaises(ValueError):
            self.arr.optimize([], [])


if __name__ == '__main__':
    unittest.main()
