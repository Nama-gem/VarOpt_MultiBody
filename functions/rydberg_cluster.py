"""Finite rectangular-array Rydberg cluster expansion. Requires only NumPy.

Convention (hbar=1): H = (Omega/2) sum_i (|r><e| + h.c.)
                         - Delta sum_i n_r,i
                         - C6 sum_{i<j} n_r,i n_r,j / r_ij**6.

Uniform dressing, isotropic van der Waals interactions, open or periodic boundaries.
Periodic interactions use minimum-image distances (not an infinite image sum).
All distances use the same units as `spacing`; all energies/frequencies use
the same units. No spatial cutoff or weak-drive approximation is made.
The implementation supports C6*Delta >= 0, the nonresonant blockade branch.
Opposite-sign parameters require explicit adiabatic eigenstate tracking and
are deliberately rejected rather than selecting a potentially wrong state.
"""

from itertools import combinations
from math import comb
import argparse
import json
import numpy as np


def canonical_geometry(points, allow_axis_swap=True):
    """Cache key invariant under translations and square-lattice symmetries."""
    points = tuple(points)
    variants = []
    for swap in ((False, True) if allow_axis_swap else (False,)):
        for sx in (-1, 1):
            for sy in (-1, 1):
                q = sorted((sx * (y if swap else x),
                            sy * (x if swap else y)) for x, y in points)
                x0, y0 = q[0]
                variants.append(tuple((x - x0, y - y0) for x, y in q))
    return min(variants)


class ClusterSolver:
    def __init__(self, omega, delta, c6, spacing=1.0, spacing_y=None, periods=None):
        spacing_y = spacing if spacing_y is None else spacing_y
        if not all(np.isfinite(v) for v in (omega, delta, c6, spacing, spacing_y)):
            raise ValueError('Parameters must be finite.')
        if delta == 0 or spacing <= 0 or spacing_y <= 0:
            raise ValueError('delta must be nonzero and spacing positive.')
        if c6 != 0 and np.sign(c6) != np.sign(delta):
            raise ValueError('This solver requires C6*Delta >= 0; see module docstring.')
        self.omega, self.delta = float(omega), float(delta)
        self.c6, self.spacing = float(c6), float(spacing)
        self.spacing_y = float(spacing_y)
        if periods is not None and (len(periods) != 2 or any(type(v) is not int or v < 1 for v in periods)):
            raise ValueError('periods must be a pair of positive integer lattice lengths.')
        self.periods = periods
        self.allow_axis_swap = spacing == spacing_y and (periods is None or periods[0] == periods[1])
        self.energy_cache = {(): 0.0}
        self.interaction_cache = {(): 0.0}

    def energy(self, points):
        """Exact cluster dressed energy on the branch connected to |ee...e>."""
        key = canonical_geometry(points, self.allow_axis_swap) if len(points) else ()
        if key in self.energy_cache:
            return self.energy_cache[key]
        m = len(key)
        states = np.arange(1 << m, dtype=np.int64)
        nr = ((states[:, None] >> np.arange(m)) & 1).astype(float)
        diagonal = -self.delta * nr.sum(axis=1)
        for i, j in combinations(range(m), 2):
            dx = key[i][0] - key[j][0]
            dy = key[i][1] - key[j][1]
            if self.periods is not None:
                dx = abs(dx) % self.periods[0]
                dy = abs(dy) % self.periods[1]
                dx = min(dx, self.periods[0]-dx)
                dy = min(dy, self.periods[1]-dy)
            r2 = self.spacing**2 * dx*dx + self.spacing_y**2 * dy*dy
            diagonal -= (self.c6 / r2**3) * nr[:, i] * nr[:, j]
        h = np.diag(diagonal)
        for i in range(m):
            h[states, states ^ (1 << i)] = self.omega / 2
        # At zero drive all excited configurations lie below (Delta>0), or
        # above (Delta<0), the vacuum. The corresponding extremal branch is
        # connected continuously to the vacuum; no overlap heuristic needed.
        eig = np.linalg.eigvalsh(h)
        e = float(eig[-1] if self.delta > 0 else eig[0])
        self.energy_cache[key] = e
        return e

    def interaction(self, points):
        """Connected occupation coefficient V_S = E_S - sum_{T proper S} V_T."""
        key = canonical_geometry(points, self.allow_axis_swap) if len(points) else ()
        if key not in self.interaction_cache:
            v = self.energy(key)
            for k in range(1, len(key)):
                for sub in combinations(key, k):
                    v -= self.interaction(sub)
            self.interaction_cache[key] = float(v)
        return self.interaction_cache[key]


def calculate_interactions(lx, ly, n, *, omega, delta, c6, spacing=1.0, spacing_y=None,
                           max_clusters=1_000_000, boundary='open'):
    """Return all occupation and spin-z coefficients through cluster order n.

    Site id = x + lx*y. Output `occupation[S]` multiplies prod_{i in S} n_i,
    with n_i=1/2+s_i^z and spin eigenvalues +/-1/2.
    Output `ising[S]` multiplies prod_{i in S} s_i^z, and includes contributions
    from every occupation cluster through order n containing S. These are
    NOT Pauli-sigma coefficients (divide by 2**len(S) to obtain those).
    `ising_constant` is the accompanying scalar energy shift.

    Every cluster is enumerated, including chain-shaped and distant clusters.
    Dense diagonalization uses dimension 2**m and roughly O(4**m) memory.
    Number of output clusters: sum(comb(lx*ly,m), m=1..n).
    max_clusters is a preallocation guard, not a physical truncation.
    """
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (lx, ly, n)):
        raise ValueError('lx, ly, and n must be integers.')
    if lx < 1 or ly < 1 or not 1 <= n <= lx*ly:
        raise ValueError('Require lx, ly >= 1 and 1 <= n <= lx*ly.')
    if boundary not in ('open', 'periodic'):
        raise ValueError('boundary must be open or periodic.')
    count = sum(comb(lx*ly, m) for m in range(1, n+1))
    if max_clusters is not None and count > max_clusters:
        raise ValueError(f'{count:,} clusters exceed max_clusters={max_clusters:,}. '
                         'Increase the limit explicitly if intended.')
    solver = ClusterSolver(omega, delta, c6, spacing, spacing_y,
                           periods=(lx, ly) if boundary == 'periodic' else None)
    sites = [(x, y) for y in range(ly) for x in range(lx)]
    occupation = {}
    for m in range(1, n+1):
        for s in combinations(range(len(sites)), m):
            occupation[s] = solver.interaction(tuple(sites[i] for i in s))

    ising = {s: 0.0 for s in occupation}
    constant = 0.0
    for s, v in occupation.items():
        m = len(s)
        constant += v / 2**m
        for k in range(1, m+1):
            for a in combinations(s, k):
                ising[a] += v / 2**(m-k)
    return dict(
        parameters=dict(lx=lx, ly=ly, n=n, omega=omega, delta=delta,
                        c6=c6, spacing=spacing, spacing_y=solver.spacing_y, boundary=boundary),
        sites=sites, occupation=occupation, ising=ising,
        ising_constant=constant,
        unique_geometries=len(solver.energy_cache)-1,
    )


def save_json(result, filename):
    """Serialize tuple-keyed interactions as records with site-id lists."""
    output = dict(result)
    for name in ('occupation', 'ising'):
        output[name] = [dict(sites=list(s), order=len(s), coefficient=v)
                        for s, v in result[name].items()]
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, allow_nan=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lx', type=int, required=True)
    p.add_argument('--ly', type=int, required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--omega', type=float, required=True)
    p.add_argument('--delta', type=float, required=True)
    p.add_argument('--c6', type=float, required=True)
    p.add_argument('--spacing', type=float, default=1.)
    p.add_argument('--spacing-y', type=float, default=None)
    p.add_argument('--boundary', choices=('open', 'periodic'), default='open')
    p.add_argument('--max-clusters', type=int, default=1_000_000)
    p.add_argument('--output', default='rydberg_interactions.json')
    args = vars(p.parse_args())
    filename = args.pop('output')
    result = calculate_interactions(**args)
    save_json(result, filename)
    print(f"Saved {len(result['occupation']):,} clusters to {filename}; "
          f"diagonalized {result['unique_geometries']:,} distinct geometries.")


if __name__ == '__main__':
    main()
