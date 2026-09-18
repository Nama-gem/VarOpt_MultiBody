import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fcntl
import numpy as np
from scipy.optimize import Bounds, OptimizeResult
from functions.generate_square_array import ConstructSquareArray
from functions.optimization_results import OptimizationResultStore


def racing_worker(directory, gate, values, barrier):
    arr = ConstructSquareArray(2, 2, 1.3, .65, 3, backend='numpy')
    store = OptimizationResultStore(directory)
    barrier.wait(timeout=30)
    for value in values:
        store.load_best(arr, [gate])  # This intentionally may become stale.
        store.save_if_better(arr, [gate], [value], value)
        record = store.load_best(arr, [gate])
        if record['parameters'] != [record['xi']]:
            raise AssertionError('Torn record observed.')


class ResultStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.arr = ConstructSquareArray(2, 2, 1.3, .65, 3, backend='numpy')
        self.store = OptimizationResultStore(self.temp.name)
        self.gates = ['Iz_echo']

    def test_strict_improvement_only(self):
        self.assertIsNone(self.store.load_best(self.arr, self.gates))
        self.assertEqual(list(Path(self.temp.name).rglob('*')), [])
        self.assertTrue(self.store.save_if_better(self.arr, self.gates, [12.], .8))
        path = self.store.path_for(self.arr, self.gates)
        before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
        for value in (.8, .9):
            self.assertFalse(self.store.save_if_better(self.arr, self.gates, [99.], value))
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino))
        self.assertTrue(self.store.save_if_better(self.arr, self.gates, [13.], .7))
        self.assertEqual(json.loads(path.read_text()), {'parameters': [13.], 'xi': .7})
        self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_case_separation_and_stable_keys(self):
        base = self.store.path_for(self.arr, self.gates)
        same = ConstructSquareArray(2, 2, 1.3, .65, 3, backend='numpy')
        self.assertEqual(base, self.store.path_for(same, self.gates))
        for attrs in [dict(_Lx=4, _Ly=1), dict(Rb=np.nextafter(1.3,2.)), dict(Omega=.7),
                      dict(n=2), dict(boundary='periodic'), dict(convention='manuscript'),
                      dict(backend='quspin'), dict(use_reflections=True)]:
            with patch.multiple(self.arr, **attrs):
                self.assertNotEqual(base, self.store.path_for(self.arr, self.gates))
        self.assertNotEqual(base, self.store.path_for(self.arr, ['XY_echo']))
        self.assertNotEqual(self.store.path_for(self.arr, ['Rx','Iz']),
                            self.store.path_for(self.arr, ['Iz','Rx']))
        self.assertNotEqual(base, self.store.path_for(self.arr, self.gates, study='other'))
        self.assertNotEqual(base, self.store.path_for(self.arr, self.gates, bounds=False))
        self.assertEqual(base, self.store.path_for(self.arr, self.gates, bounds=Bounds([0],[np.inf])))
        self.assertEqual(base, self.store.path_for(self.arr, self.gates, bounds=[(0,None)]))
        with patch.object(self.arr, '_initial_state', -self.arr._initial_state):
            self.assertNotEqual(base, self.store.path_for(self.arr, self.gates))

    def test_corruption_and_bad_values_do_not_overwrite(self):
        self.store.save_if_better(self.arr, self.gates, [12.], .8)
        path = self.store.path_for(self.arr, self.gates)
        for bad in (np.nan, np.inf, -.1):
            with self.assertRaises(ValueError):
                self.store.save_if_better(self.arr, self.gates, [12.], bad)
        with self.assertRaises(ValueError):
            self.store.save_if_better(self.arr, self.gates, [np.nan], .5)
        with self.assertRaises(ValueError):
            self.store.save_if_better(self.arr, self.gates, [1., 2.], .5)
        path.write_text('{broken')
        with self.assertRaises(ValueError):
            self.store.save_if_better(self.arr, self.gates, [1.], .1)
        with self.assertRaises(ValueError):
            self.store.load_best(self.arr, self.gates)
        self.assertEqual(path.read_text(), '{broken')

    def test_replace_failure_preserves_old_record(self):
        self.store.save_if_better(self.arr, self.gates, [12.], .8)
        path = self.store.path_for(self.arr, self.gates)
        with patch('functions.optimization_results.os.replace', side_effect=OSError('injected')):
            with self.assertRaises(OSError):
                self.store.save_if_better(self.arr, self.gates, [13.], .7)
        self.assertEqual(self.store.load_best(self.arr, self.gates)['xi'], .8)
        self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_lock_timeout_does_not_act_as_missing_record(self):
        self.store.save_if_better(self.arr, self.gates, [12.], .8)
        path = self.store.path_for(self.arr, self.gates)
        store = OptimizationResultStore(self.temp.name, lock_timeout=0)
        with Path(str(path)+'.lock').open('a+b') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                with self.assertRaises(TimeoutError):
                    store.save_if_better(self.arr, self.gates, [13.], .7)
                with self.assertRaises(TimeoutError):
                    store.load_best(self.arr, self.gates)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        self.assertEqual(store.load_best(self.arr, self.gates)['xi'], .8)

    def test_multiple_processes_keep_minimum_and_matching_parameters(self):
        context = mp.get_context('spawn')
        barrier = context.Barrier(4)
        processes = [context.Process(target=racing_worker, args=(self.temp.name, 'Iz_echo', values, barrier))
                     for values in ([.9,.5,.8], [.7,.4,.6], [.6,.2,.9], [.8,.1,.3])]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=45)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()
        self.assertEqual(self.store.load_best(self.arr, self.gates), {'parameters':[.1], 'xi':.1})

    def test_optimize_integration(self):
        result = self.arr.optimize(self.gates, [10.], results_dir=self.temp.name,
                                    bounds=[(1,150)], options={'maxiter':100})
        self.assertTrue(result.success)
        self.assertTrue(result.saved)
        self.assertEqual(self.store.load_best(self.arr, self.gates, bounds=[(1,150)]),
                         {'parameters': result.x.tolist(), 'xi': result.fun})
        self.assertEqual(result.result_path, str(self.store.path_for(self.arr, self.gates, bounds=[(1,150)])))
        failed = OptimizeResult(x=np.array([1.]), fun=.01, success=False)
        with patch('functions.generate_square_array.minimize', return_value=failed):
            result = self.arr.optimize(self.gates, [10.], results_dir=self.temp.name, bounds=[(1,150)])
        self.assertFalse(result.saved)
        self.assertGreater(self.store.load_best(self.arr, self.gates, bounds=[(1,150)])['xi'], .01)
        with self.assertRaisesRegex(ValueError, 'study'):
            self.arr.optimize(self.gates, [10.], results_dir=self.temp.name,
                              constraints={'type':'ineq', 'fun':lambda x:150-x[0]})


if __name__ == '__main__':
    unittest.main()
