"""Run independent workers against one best-result directory.

Start several copies with the same system options and --results-dir, using
independent random seeds. Each worker re-reads the latest best before a new
attempt; saving compares again under a lock, never using that stale snapshot.
"""
import argparse
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from functions.generate_square_array import ConstructSquareArray
from functions.optimization_results import OptimizationResultStore, DEFAULT_RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lx', type=int, default=4)
    parser.add_argument('--ly', type=int, default=4)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--rb', type=float, default=1.5)
    parser.add_argument('--omega', type=float, default=.5)
    parser.add_argument('--boundary', choices=('open', 'periodic'), default='open')
    parser.add_argument('--interaction', choices=('Iz', 'Iz_echo', 'XY', 'XY_echo'), default='Iz_echo')
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--hessian', action='store_true')
    args = parser.parse_args()
    if args.attempts < 1 or args.maxiter < 1:
        parser.error('--attempts and --maxiter must be positive')
    arr = ConstructSquareArray(args.lx, args.ly, args.rb, args.omega, args.n, boundary=args.boundary)
    store = OptimizationResultStore(args.results_dir)
    gates = [args.interaction, 'Rx', args.interaction]
    bounds = [(1., 200.), (-np.pi, np.pi), (1., 200.)]
    low, high = np.array(bounds).T
    rng = np.random.default_rng(args.seed)
    print('Shared best record:', store.path_for(arr, gates, bounds=bounds), flush=True)
    for attempt in range(args.attempts):
        best = store.load_best(arr, gates, bounds=bounds)
        if best is not None and rng.random() < .5:
            initial = np.clip(np.asarray(best['parameters']) + .1*(high-low)*rng.normal(size=len(gates)), low, high)
        else:
            initial = rng.uniform(low, high)
        result = arr.optimize(gates, initial, bounds=bounds, hessian=args.hessian,
                              options={'maxiter':args.maxiter}, results_dir=store.directory)
        print(f'worker={os.getpid()} attempt={attempt+1} xi={result.fun:.10g} '
              f'success={result.success} saved={result.saved}', flush=True)


if __name__ == '__main__':
    main()
