"""Run directly for a small example; --lx 5 --ly 4 --n 3 for 20 spins."""
import argparse
import time
import os
# Avoid excessive OpenMP overhead on these small sparse problems; a user-set
# value takes precedence. Set before importing NumPy/QuSpin.
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from functions.generate_square_array import ConstructSquareArray


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lx', type=int, default=3)
    p.add_argument('--ly', type=int, default=2)
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--rb', type=float, default=1.5)
    p.add_argument('--omega', type=float, default=.5)
    p.add_argument('--optimize', action='store_true')
    p.add_argument('--hessian', action='store_true', help='Use exact Hessian and trust-constr with --optimize')
    p.add_argument('--boundary', choices=('open', 'periodic'), default='open')
    p.add_argument('--backend', choices=('quspin', 'numpy'), default='quspin')
    p.add_argument('--no-reflections', action='store_true')
    p.add_argument('--interaction', choices=('Iz', 'Iz_echo', 'XY', 'XY_echo'), default='Iz_echo')
    args = p.parse_args()
    start = time.perf_counter()
    arr = ConstructSquareArray(args.lx, args.ly, args.rb, args.omega, args.n,
                               boundary=args.boundary, backend=args.backend,
                               use_reflections=not args.no_reflections)
    print(f'{arr._L} spins, {arr.dim:,}/{arr.full_dim:,} amplitudes, {len(arr.occupation):,} clusters')
    print(f'{args.boundary} boundaries, {args.backend} backend, reflections={arr.use_reflections}')
    print(f'Construction: {time.perf_counter()-start:.3f} s')
    gates = [args.interaction, 'Rx', args.interaction]
    theta = [20., .4, 15.]
    start = time.perf_counter()
    psi = arr.run_sequence(theta, gates)
    xi = arr.squeezing(psi)
    print(f'Sequence + squeezing: {time.perf_counter()-start:.3f} s')
    print(f'Norm={np.linalg.norm(psi):.12f}, xi={xi:.8f}, xi^2={xi**2:.8f}')
    if args.optimize:
        result = arr.optimize(gates, theta, bounds=[(0, 200), (-np.pi, np.pi), (0, 200)],
                              hessian=args.hessian, options={'maxiter': 100})
        print(result)


if __name__ == '__main__':
    main()
