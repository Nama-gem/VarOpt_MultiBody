# VarOpt_MultiBody_Ising

Static many-body Rydberg-dressed Ising evolution with global rotations,
analytic gradients and an optional exact objective Hessian. Physical inputs
are `Rb = Rb/a` and `Omega = Omega/Delta`; `Lx`, `Ly` and `n` set the geometry
and connected-cluster truncation. There are no ramps.

```python
from functions.generate_square_array import ConstructSquareArray
from functions import dressing_functions_spin as df

arr = ConstructSquareArray(4, 4, Rb=1.5, Omega=0.5, n=3,
                           boundary='open', backend='quspin',
                           use_reflections=True)
# Use boundary='periodic' for a torus.
gates = ['Iz_echo', 'Rx', 'Iz_echo']
theta = [20.0, 0.4, 15.0]
psi, dpsi = arr.evaluate_gate_sequence(theta, gates, gradient=True)
xi, dxi = df.evaluate_gate_sequence_squeezing(theta, gates, arr, gradient=True)

# Exact-Hessian optimization, with bounds on times and rotation angles:
result = arr.optimize(gates, theta, hessian=True,
                      bounds=[(1, 200), (-3.14159, 3.14159), (1, 200)],
                      options={'maxiter': 100, 'gtol': 1e-8})
```

The class is also exported as `VarOptMultiBodyIsing`. The function-style
squeezing interface follows `VarOpt_SoftCore_Ising_Torus`; dense operator
diagonalizations are not used here. `Ix` below uses the full many-body
interaction coefficients, rather than the old soft-core pair potential.
`XY` and `XY_echo` use the many-body-derived pair coefficients described below.

## Project layout and notebook

```text
VarOpt_MultiBody_Ising/
  example_optimization.ipynb     # runnable 4x4 optimization walkthrough
  functions/                    # importable simulation and storage package
  tests/                        # all regression tests
  example_sequence.py           # command-line examples
  example_optimization_worker.py
  data/optimization_results/    # created when saving results
  requirements.txt
  requirements-notebook.txt
```

Start with `example_optimization.ipynb` in the main folder and select the
project's `.venv` Python interpreter as its notebook kernel. Run all cells in
order. The example uses a 4x4 open array with n=3 and one bounded optimization.
Its executed output is included; saving is disabled by default in the example.
Install notebook dependencies with `python -m pip install -r requirements-notebook.txt`
when setting up a new environment.

Imports now use the package, for example `from functions import ConstructSquareArray`
or `from functions import dressing_functions_spin as df`. Existing external scripts
that used top-level imports must be updated likewise. Run tests from the main
folder using `python -m unittest discover -s tests -v` (plain discovery also works).
The saved-result directory and case identities are unchanged by the move.

## Hilbert space and symmetries

The default `backend='quspin'` constructs a QuSpin `spin_basis_general` basis.
With `use_reflections=True` (default), it selects the **even spatial-reflection
sectors along x and y**. Both reflections preserve the uniform interaction,
all supported gates, and |+x>^N, for either boundary condition. The two
reflections commute. No total-magnetization or spin-inversion sector is fixed:
odd-z interactions and arbitrary global rotations remain supported.
Translations and diagonal reflections are not currently included.

For a **4x4 array, this reduces 65,536 amplitudes to 16,576**. `arr.dim` is the
working dimension, `arr.full_dim` is 2**N, and `arr.basis` is the QuSpin basis.
Collective spin operators use `pauli=0`, i.e. J=sum sigma/2, and are stored as
CSR matrices in `arr.op['Jx']`, `arr.op['Jy']`, `arr.op['Jz']`. Rotations use
sparse exponential multiplication. Ising evolution remains elementwise phase
multiplication on a **diagonal Hamiltonian even in the reduced basis**.

`backend='numpy'` retains the original full computational basis and tensor
product rotations; reflections are then unused. For an unreduced QuSpin basis,
set `use_reflections=False`.

**State shapes now follow `arr.dim`, not necessarily `2**N`.** QuSpin uses its
native reduced ordering; use the explicit conversion helpers:

```python
psi_full = arr.to_full(psi)             # (2**N, 1), ascending bit-i order
psi_reduced = arr.from_full(psi_full)   # (arr.dim, 1)
dpsi_full = arr.to_full(dpsi.T).T       # (len(gates), 2**N)
```

Full computational site id is `x + Lx*y`; bit i=1 means sigma_z,i=+1.
The helpers handle QuSpin's site-bit and row ordering explicitly.
`from_full` rejects states outside the selected reflection sectors rather
than silently changing the state. To simulate an arbitrary nonsymmetric state,
use `use_reflections=False` or the NumPy backend. Projection helpers accept
multiple columns; `run_sequence(state=...)` expects one normalized state in
the working basis. The default initial state is |+x>^N.

## Boundaries and interaction conventions

- `boundary='open'`: ordinary Euclidean pair distances.
- `boundary='periodic'`: minimum-image distances along each dimension,
  `dx=min(abs(xi-xj), Lx-abs(xi-xj))`, likewise for y. Each pair is counted once.
  This is a finite torus, **not an infinite periodic-image sum**.

The boundary choice enters each microscopic cluster Hamiltonian, before
connected coefficients are computed; it changes the physical interactions
as well as the interpretation of the array. Cluster caching respects unequal
period lengths and does not incorrectly interchange rectangular axes.

We set `a = Delta = hbar = 1` with positive detuning on the nonresonant
blockade branch. `C6/Delta/a^6 = 2*(Rb/a)^6`. Default `convention='paper'`
uses the positive-detuning, positive-interaction convention from the recent
`Many-body Interactions Dressing` examples:

    H_Ryd/Delta = (Omega/Delta)/2 sum X_er + sum n_r
                 + 2*(Rb/a)^6 sum n_r,i n_r,j / (r_ij/a)^6.

For each subset S, select the eigenenergy connected to all dressed spin
levels at zero drive, then obtain connected coefficients V_S by
inclusion-exclusion. Retain **every occupation cluster of size <= n**, including
distant clusters. Then

    H_Ising/Delta = sum_{1 <= |S| <= n} V_S prod_{i in S} n_i,
    n_i = (1 + sigma_z,i)/2.

`occupation` stores V_S. `pauli_z` stores coefficients of products of Pauli
sigma_z, and `constant` is the scalar term. `pauli_z_even` contains nonconstant
even-z terms, including contributions from all retained occupation clusters.
`h_ising` and `h_echo` are diagonals in the working basis, including constant
and single-particle shifts as appropriate.

`convention='manuscript'` reverses the microscopic diagonal signs, matching
the older minus-sign solver. This is a convention switch, not another physical
variational parameter. Resonant/attractive branches are outside this model.
Times are **Delta*t_phys**, without nearest-neighbor or soft-core plateau
normalization. Changing Delta at fixed ratios only rescales physical time.

## Gates and first derivatives

One independent parameter per gate, acting in list order:

| Gate | Parameter | Operation |
|---|---|---|
| `Rx`, `Ry`, `Rz` | radians | exp(-i angle sum sigma_axis/2) |
| `Iz` | total time | exp(-i time H_Ising) |
| `Iz_echo` | total time | exp(-i time H_even), spin frame restored |
| `Ix` | total time | exp(-i time Hx), every z-product replaced by an x-product |
| `Ix_echo` | total time | exp(-i time Hx_even), only even x-products retained |
| `Dz` | total time | U(time/2), Rx(pi), U(time/2), final pi rotation retained |
| `XY` | total time | exp(-i time H_XY), replacing pair zz by xx+yy |
| `XY_echo` | total time | exp(-i time H_XY_even), odd z terms removed |

For P=product sigma_x, `H_even=(H_Ising+P H_Ising P)/2` exactly removes odd
Pauli-z terms. `Iz_echo` equals U(t/2), Rx(pi), U(t/2), Rx(-pi). `Dz` retains
Rx(pi), including its global phase. Odd-sized occupation clusters must NOT
be removed before expanding them into Pauli terms.

## X-axis many-body Ising interactions

`Ix` rotates the **entire** Ising Hamiltonian to the x-axis, including
one-body and higher-body terms, with the same coefficients:

    Hx = constant + sum_S J_S prod_{i in S} X_i.
    Hx_even = constant + sum_{|S| even} J_S prod_{i in S} X_i.

`Ix_echo` evolves under Hx_even with no residual pi rotation. Odd occupation
clusters can still contribute to its even Pauli coefficients; the filtering
is performed after the occupation-to-Pauli conversion, as for `Iz_echo`.

Implementation uses the exact conjugation
`Hx = Ry(pi/2) Hz Ry(-pi/2)`: apply Ry(-pi/2), then diagonal Ising phases,
then Ry(pi/2). This is not a Trotter approximation and does not require
constructing or diagonalizing a new many-spin Hamiltonian. The generator
is applied by the same conjugation for analytic gradients and Hessians.
All spatial reflections and both boundary conditions remain supported.

Because all x-products commute, `Ix_echo(t)` also equals the physical
sequence Ix(t/2), Rz(pi), Ix(t/2), Rz(-pi), including frame restoration.

```python
gates = ['Iz_echo', 'Rx', 'Ix_echo']
theta = [20., .4, 15.]
psi, dpsi = arr.evaluate_gate_sequence(theta, gates, gradient=True)
xi, dxi, Hxi = arr.evaluate_gate_sequence_squeezing(theta, gates, hessian=True)

# Direct interfaces:
psi = arr.evolve_x(state, time, echo=True)
Hpsi = arr.apply_ising_x(state, echo=True)
# Function-style evolution: df.apply_H_Ix and df.apply_H_Ix_echo.
```

The default |+x>^N initial state is an eigenstate of both Hx and Hx_even.
Consequently `Ix` or `Ix_echo` alone only gives that state a global phase;
interleave it with a rotation away from x or another interaction such as Iz
to obtain nontrivial dynamics. An all-Ix sequence on the default initial state
does not squeeze and has the same degenerate transverse variances that make
the squeezing Hessian undefined there.

## XY interactions

`XY` performs the literal pair replacement

    J_ij sigma_z,i sigma_z,j -> J_ij (sigma_x,i sigma_x,j + sigma_y,i sigma_y,j).

There is **no additional factor of 1/2**. For example, the exchange matrix
element between |01> and |10> is 2*J_ij. J_ij is taken from `pauli_z[(i,j)]`,
so it includes contributions from every retained occupation cluster, not
just the two-atom cluster calculation. All other terms are kept unchanged:

    H_XY = constant + sum_i J_i Z_i
           + sum_{i<j} J_ij (X_i X_j + Y_i Y_j)
           + sum_{|S|>=3} J_S prod_{i in S} Z_i.

`XY_echo` removes odd z products and retains the same exchange:

    H_XY_even = constant + sum_{i<j} J_ij (X_i X_j + Y_i Y_j)
                + sum_{|S|>=4, |S| even} J_S prod_{i in S} Z_i.

Thus at n=2 or n=3, `XY_echo` contains only exchange and a scalar constant;
the pair coefficients still depend on the chosen n. At n>=4, even higher-body
z terms are retained. `XY` and `XY_echo` preserve total Jz and the spatial
reflections used by the basis.

For P=product sigma_x, `H_XY_even=(H_XY+P H_XY P)/2`. This is an **effective
parity-filtered Hamiltonian**: unlike diagonal Ising, a single physical
midpoint pi pulse generally does NOT give exp(-it H_XY_even), because the
exchange can fail to commute with the odd-z remainder. `XY_echo` implements
the filtered Hamiltonian itself, without a residual pi rotation, and does
not simulate a finite echo-pulse sequence.

```python
gates = ['XY_echo', 'Rx', 'XY']
theta = [20.0, 0.4, 15.0]
xi, dxi, Hxi = arr.evaluate_gate_sequence_squeezing(theta, gates, hessian=True)
psi, dpsi = arr.evaluate_gate_sequence(theta, gates, gradient=True)

# Direct evolution or generator access, in the working basis:
psi = arr.evolve_xy(arr._initial_state, 20.0, echo=True)
Hxy = arr.xy_operator(echo=True)  # Hermitian scipy LinearOperator
Hpsi = Hxy @ psi
# Function-style helpers: df.apply_H_XY and df.apply_H_XY_echo.
```

Unlike `Iz`, XY evolution is not diagonal. It uses exponential multiplication
without constructing a dense unitary or using a Trotter approximation. The
QuSpin backend constructs one real sparse exchange matrix on first XY use
and reuses it for both variants; their residual z terms remain diagonal.
The NumPy backend applies exchange through bit-block swaps without storing
a full sparse Hamiltonian. Both expose a LinearOperator and its exact trace
to the exponential routine. Gradients and Hessians use the same generator,
and both optimizers accept the new gate names. XY evolution is generally
more expensive than diagonal Ising evolution, particularly at 20 spins.

## Evaluating sequences

```python
psi, dpsi = arr.evaluate_gate_sequence(theta, gates, gradient=True)
# psi: (arr.dim,), dpsi[k,b] = d psi[b]/d theta[k]
xi, dxi, psi, dpsi = arr.evaluate_gate_sequence_squeezing(
    theta, gates, gradient=True, return_state=True)
```

State gradients use `dU/dtheta=-i G U`, propagated through all subsequent
gates. They differentiate gate angles/times, holding the initial state,
Rb, Omega and n fixed. No finite differences enter simulation or optimization.
`run_sequence` also supports a zero-length sequence.

Squeezing returns **xi**, not xi squared, with the full centered spin covariance
minimized perpendicular to the actual mean spin. Convert to dB using
`20*log10(xi)`. At zero mean spin xi is infinite and the gradient undefined.
At degenerate transverse variances, first-order evaluation selects one branch.

## Analytic Hessian

```python
xi, dxi, Hxi = arr.evaluate_gate_sequence_squeezing(theta, gates, hessian=True)
# Hxi[i,j] = d2 xi / (d theta[i] d theta[j]), shape (len(gates), len(gates))
xi, dxi, Hxi, psi, dpsi = df.evaluate_gate_sequence_squeezing(
    theta, gates, arr, hessian=True, return_state=True)
```

`hessian=True` implies first derivatives too. It propagates unique mixed
state derivatives and uses `d2U/dtheta2=-G**2 U` for repeated derivatives.
Only the p*(p+1)/2 unique second-derivative vectors are stored. The objective
Hessian differentiates the spin moments, covariance centering, mean-spin
normalization, transverse frame and minimum transverse eigenvalue. It is
analytic up to floating-point precision for the chosen truncated model.

At vanishing mean spin, zero variance, or a degenerate/ill-conditioned transverse
eigenvalue gap, Hessian evaluation raises a descriptive ValueError. In
particular, do not initialize it at an all-zero sequence acting on the coherent
state. Use a nonzero interacting sequence or refine a gradient-only result.

## Optimization controls

```python
result = arr.optimize(
    gates, theta,
    method='L-BFGS-B',
    bounds=[(1, 200), (-3.14159, 3.14159), (1, 200)],
    gradient=True,
    hessian=False,
    tol=1e-8,
    options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 40},
    callback=lambda x: print('Current parameters:', x),
)

# Bounded exact-Hessian optimization:
result = arr.optimize(gates, theta, method='trust-constr', hessian=True,
                      options={'maxiter': 200, 'gtol': 1e-8, 'verbose': 1})

# Derivative-free optimization (no state gradients computed):
result = arr.optimize(gates, theta, method='Powell', options={'maxiter': 300})

# Unconstrained optimization: bounds=False explicitly permits any real times.
result = arr.optimize(gates, theta, method='BFGS', bounds=False)
result = arr.optimize(gates, theta, method='trust-exact', bounds=False)
```

| Argument | Meaning |
|---|---|
| `method` | Standard SciPy solver name, case-insensitive; choices below |
| `bounds` | None preserves default bounds; a pair list or SciPy Bounds overrides them; False removes all bounds |
| `gradient` | None automatically supplies analytic gradients where supported; True explicitly requests them; False permits numerical gradients where the solver supports them |
| `hessian` | None automatically supplies the exact Hessian for Newton-CG and the unconstrained Newton trust-region methods; True explicitly requests it; False disables it |
| `tol` | SciPy's general stopping tolerance |
| `options` | Method-specific controls, e.g. maximum iterations and separate convergence tolerances |
| `callback` | A SciPy iteration callback, forwarded unchanged |
| `constraints` | SciPy constraint objects or dictionaries, forwarded unchanged |

Without `method`, the default remains **L-BFGS-B** with analytic gradients.
`hessian=True` selects **trust-constr**. Providing constraints without an exact
Hessian selects **SLSQP**. Default bounds remain [-pi, pi] for rotations and
[0, infinity) for interaction times. No supplied bounds, constraints, or
explicit derivative requests are silently ignored: incompatible combinations
raise ValueError before running. For an unconstrained method, explicitly pass
`bounds=False`; such a run can choose negative interaction times.

Supported methods and derivative behavior:

| Methods | Bounds | General constraints | Derivatives |
|---|---|---|---|
| L-BFGS-B, TNC | Yes | No | Analytic gradient by default |
| SLSQP | Yes | Yes | Analytic gradient by default |
| Powell, Nelder-Mead | Yes | No | No gradients |
| COBYLA, COBYQA | Yes | Yes, solver-dependent constraint types | No gradients |
| BFGS, CG | No | No | Analytic gradient by default |
| Newton-CG | No | No | Analytic gradient and exact Hessian by default; hessian=False uses SciPy's approximation |
| dogleg, trust-exact, trust-ncg, trust-krylov | No | No | Analytic gradient and exact Hessian required |
| trust-constr | Yes | Yes | Analytic gradient by default; exact Hessian only when hessian=True |

`dogleg` requires a positive-definite Hessian, which need not hold far from
a minimum. Gradient-free methods bypass derivative calculations even when
`gradient` is omitted. `gradient=False` on e.g. BFGS uses SciPy finite
differences of the objective; the explicit exact-Hessian option requires
analytic gradients as well. Value, gradient and Hessian share a cached
evaluation. Returned objects remain SciPy OptimizeResult (`x`, `fun`,
`success`, `message`, etc.). All these methods are local optimizers.

For example, constrain the total interaction time in a three-gate sequence:

```python
from scipy.optimize import LinearConstraint
time_budget = LinearConstraint([[1, 0, 1]], 0, 150)
result = arr.optimize(gates, theta, method='SLSQP', constraints=[time_budget],
                      options={'maxiter': 300, 'ftol': 1e-9})
```

Options and callback signatures follow
[SciPy minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
For instance, trust-constr accepts `maxiter`, `gtol`, `xtol`, `barrier_tol`,
and `verbose`; L-BFGS-B accepts `maxiter`, `gtol`, `ftol`, and `maxls`; TNC
uses `maxfun` instead of `maxiter`. `callback(xk)` works for most methods.
For trust-constr use `callback(xk, state)` or the named argument
`callback(intermediate_result)`; the latter also works with most other
methods. Consult the selected solver's documentation for stopping behavior.

## Saving best results across workers

```python
from functions.optimization_results import OptimizationResultStore

store = OptimizationResultStore()  # project/data/optimization_results
gates = ['Iz_echo', 'Rx', 'Iz_echo']
bounds = [(1, 200), (-3.14159, 3.14159), (1, 200)]

best = store.load_best(arr, gates, bounds=bounds)
theta0 = best['parameters'] if best is not None else [20., .4, 15.]
result = arr.optimize(gates, theta0, bounds=bounds,
                      results_dir=store.directory,
                      options={'maxiter': 300})
print(result.saved)        # True only if this run installed a new best record
print(result.result_path)  # exact path for this case

best = store.load_best(arr, gates, bounds=bounds)
# best is None if absent, otherwise {'parameters': [...], 'xi': ...}
```

Saving is opt-in: `results_dir=None` (the default) leaves optimization purely
in memory. `OptimizationResultStore()` defaults to the project-root data directory, independent of the current working directory. Pass an
explicit shared directory for multiple workers, e.g.
`OptimizationResultStore('/shared/project/optimization_results')`.

**System identification includes array size, Rb/a and Omega/Delta.** It also
includes cluster order n, boundary condition, sign convention, the ordered gate
sequence, parameter bounds, backend/reflection settings, an initial-state
fingerprint, and a model/objective version. Different solvers, seeds, starting
points and tolerances compete for the same record. The system information is
encoded in the directory and filename; each JSON contains only:

```json
{"parameters":[20.0,0.4,15.0],"xi":0.9}
```

The example numbers above illustrate the format, not a computed optimum.
`xi` is the squeezing parameter itself, not xi squared or dB. There are no
stored wavefunctions, gradients, histories, timestamps or optimizer metadata.
A typical record lives under:

```text
data/optimization_results/
  4x4_n3_open_paper/
    Rb1.5_Omega0.5/
      default/
        Iz_echo_Rx_Iz_echo--<case-hash>.json
```

The hash includes the full-precision physical parameters and bounds, so nearby
values that round to the same readable directory name remain separate. Use
`store.path_for(arr, gates, bounds=bounds)` to locate a record; it does not
create files. Load using the same bounds used during optimization: omitted
bounds mean the default ranges, and `bounds=False` means unconstrained.
Equivalent list and scipy Bounds representations use the same key.

On completion, `arr.optimize(..., results_dir=...)` automatically submits only
**successful final results with finite nonnegative xi and finite parameters**.
Unsuccessful runs (including iteration-limit termination) are not automatically
saved. They still return normally, with `result.saved=False`. This avoids
silently accepting an infeasible or failed constrained run. To deliberately
submit a separately validated candidate, use the lower-level API:

```python
improved = store.save_if_better(arr, gates, parameters, xi, bounds=bounds)
```

That method validates numerical values but trusts the supplied objective and
the candidate's feasibility; it does not rerun evolution or custom constraints.
It returns True for a first record or a strictly lower xi, False for a tie or
worse result. Exact ties preserve the earlier parameters.

For custom constraints, saving through `optimize` requires a stable `study`
label identifying the problem, e.g. `study='total_time_le_150'`. Pass that same
label to `load_best` and `save_if_better`. Change it when changing custom
constraints: arbitrary Python constraint functions are not serialized or
fingerprinted. Optional study labels can also separate independent campaigns.

### Concurrent updates

Every update holds an exclusive lock on a stable `.json.lock` sidecar while
it reads the latest result, compares xi, and atomically replaces the JSON only
if improved. Reading a result before optimization is not enough: the latest
comparison always happens again under the write lock. Consequently a slower
worker cannot overwrite a newer, better result. Equal/worse submissions do
not rewrite the JSON or change its modification time.

New JSON is written to a temporary file in the same directory, flushed and
fsynced, then atomically renamed; a failed replacement preserves the old
record. Lock timeouts, malformed existing files, and I/O errors raise rather
than treating the old result as missing and overwriting it. The default lock
timeout is 30 seconds; configure it on `OptimizationResultStore` if needed.
Locks are released by the OS when a process exits. Sidecar files remain;
do not delete them while workers are running.

This implementation uses POSIX flock on macOS/Linux. Workers must cooperate
through this API and share a filesystem with working flock and atomic rename.
Ordinary cloud-synced copies on separate computers do not provide that shared
lock. Check these guarantees for a cluster's shared filesystem.

`example_optimization_worker.py` repeatedly reads the best record, chooses a
random start or perturbs the best parameters, optimizes, and submits successful
improvements. Run copies in separate terminals or scheduler jobs with the same
system options, directory and bounds; independent seeds explore different starts:

```sh
python example_optimization_worker.py --results-dir /shared/results --seed 1 --attempts 20
python example_optimization_worker.py --results-dir /shared/results --seed 2 --attempts 20
```

The script does not reserve cases: multiple workers may intentionally search
the same case simultaneously. It stores no extra per-worker data files.

## Running and scaling

```sh
python -m pip install -r requirements.txt
python example_sequence.py
python example_sequence.py --lx 4 --ly 4 --n 3 --optimize --hessian
python example_sequence.py --lx 4 --ly 4 --boundary periodic --optimize --hessian
python example_sequence.py --lx 5 --ly 4 --n 3
python example_sequence.py --lx 4 --ly 4 --interaction XY_echo --optimize --hessian
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The example defaults OpenMP to one thread unless you explicitly set it; many
threads can add overhead to small sparse calculations. NumPy, SciPy and QuSpin
are installed in the project's `.venv`. The bundled `rydberg_cluster.py` is
adapted from the existing many-body project to support both boundary types.
There is no dependency on that project's paths or ramp modules.

At 4x4 with both reflections, each complex state occupies about 0.253 MiB.
For ten gate parameters, the 55 second-derivative states occupy about 13.9 MiB,
plus the initial/first derivative states and work buffers. No dense many-spin
Hamiltonian is formed. Construction currently builds full diagonals and a
sparse projector before restricting to the QuSpin basis; subsequent evolution
and derivative propagation operate entirely in the working basis.

Increasing n affects cluster construction cost, not the size of the spin basis
or the number of gate derivatives. Construction still enumerates all clusters,
and each microscopic m-site cluster uses a dense 2**m matrix. Start with n=2
or 3 and test convergence. Default guards are 20 sites and one million clusters.
Higher cluster orders can be expensive even when time evolution is fast.

`benchmark_4x4.json` records example runs for both boundaries with n=3,
Rb=1.5, Omega=0.5, gates [Iz_echo, Rx, Iz_echo], initial parameters
[20, 0.4, 15], bounds [(1,200),(-pi,pi),(1,200)], maxiter=100 and gtol=1e-7
(L-BFGS-B also used ftol=1e-12). OpenMP and OpenBLAS each used one thread.
Both solvers converged, but trust-constr was slower for these three-parameter
examples and reached a different local minimum in the open case. This is
a smoke benchmark, not evidence of a general speed advantage for either method.
