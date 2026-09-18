"""Static many-body Ising simulator with open or minimum-image periodic boundaries.

Units: a=Delta=hbar=1, Rb=(C6/(2 Delta))**(1/6). The default
paper convention is +Delta n_r + C6 n_r n_r/r**6 + Omega X/2.
Computational bit i=1 means sigma_z,i=+1 (the dressed spin state).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply, LinearOperator
from .rydberg_cluster import calculate_interactions
from .second_order import squeezing_second_order


class ConstructSquareArray:
    """QuSpin reflection-reduced simulator, with an optional NumPy full basis.

    Rb is Rb/a, Omega is Omega/Delta, n is the maximum connected
    occupation-cluster size. No distance cutoff or ramps. boundary='periodic'
    uses minimum-image distances; boundary='open' is the default.
    backend='quspin' defaults to both even spatial-reflection sectors.
    backend='numpy' retains the full computational basis.
    Times are Delta*t_phys, rotation angles are radians. Parameters are fixed
    at construction: build a new object when changing the interaction.
    """
    gates = ('Rx', 'Ry', 'Rz', 'Iz', 'Iz_echo', 'Dz', 'Ix', 'Ix_echo', 'XY', 'XY_echo')

    def __init__(self, Lx, Ly, Rb, Omega, n, *, max_sites=20,
                 max_clusters=1_000_000, convention='paper', boundary='open',
                 backend='quspin', use_reflections=True):
        if any(type(v) is not int or v < 1 for v in (Lx, Ly, n, max_sites)):
            raise ValueError('Lx, Ly, n and max_sites must be positive integers.')
        if n > Lx*Ly or Lx*Ly > max_sites:
            raise ValueError('Require n <= Lx*Ly <= max_sites (default 20).')
        if not np.isfinite(Rb) or Rb < 0 or not np.isfinite(Omega):
            raise ValueError('Rb must be finite and nonnegative; Omega must be finite.')
        if convention not in ('paper', 'manuscript'):
            raise ValueError('convention must be paper or manuscript.')
        if boundary not in ('open', 'periodic'):
            raise ValueError('boundary must be open or periodic.')
        if backend not in ('numpy', 'quspin'):
            raise ValueError('backend must be numpy or quspin.')
        self._Lx, self._Ly, self._L = Lx, Ly, Lx*Ly
        self.Rb, self.Omega, self.n = float(Rb), float(Omega), n
        self.convention = convention
        self.boundary, self.backend = boundary, backend
        self.use_reflections = bool(use_reflections) if backend == 'quspin' else False
        self.dim = 1 << self._L
        self.full_dim = self.dim
        sign = -1 if convention == 'paper' else 1
        self.interactions = calculate_interactions(
            Lx, Ly, n, omega=Omega, delta=sign, c6=sign*2*Rb**6,
            max_clusters=max_clusters, boundary=boundary)
        self.occupation = self.interactions['occupation']
        self.pauli_z = {s: v / 2**len(s) for s, v in self.interactions['ising'].items()}
        self.pauli_z_even = {s: v for s, v in self.pauli_z.items() if len(s) % 2 == 0}
        self.constant = self.interactions['ising_constant']
        # Subset zeta transform: E[b] = sum_{S subset b} V[S].
        # This avoids one million entries times thousands of cluster terms.
        self.h_ising = np.zeros(self.dim)
        for sites, coefficient in self.occupation.items():
            self.h_ising[sum(1 << i for i in sites)] = coefficient
        for i in range(self._L):
            block = self.h_ising.reshape(-1, 2, 1 << i)
            block[:, 1] += block[:, 0]
        self.h_echo = (self.h_ising + self.h_ising[::-1]) / 2
        self._jz = np.zeros(self.dim)
        for i in range(self._L):
            self._jz.reshape(-1, 2, 1 << i)[:, 1] += 1
        self._jz -= self._L / 2
        self._initial_state = np.ones((self.dim, 1), complex) / np.sqrt(self.dim)
        self._ground_state = np.zeros((self.dim, 1), complex)
        self._ground_state[0] = 1
        if backend == 'quspin':
            self._build_quspin_basis()
        # XY is optional and more costly to construct than diagonal Ising.
        # Share one exchange operator between its bare and echoed variants.
        self._xy_operators = None

    def _build_quspin_basis(self):
        """Even x/y spatial-reflection sectors; no magnetization or spin parity restriction."""
        try:
            from quspin.basis import spin_basis_general
            from quspin.operators import hamiltonian
        except ImportError as exc:
            raise ImportError('Install quspin, or select backend="numpy" for the full basis.') from exc
        sites = np.arange(self._L)
        x, y = sites % self._Lx, sites // self._Lx
        blocks = {}
        if self.use_reflections:
            if self._Lx > 1:
                blocks['reflection_x'] = ((self._Lx-1-x)+self._Lx*y, 0)
            if self._Ly > 1:
                blocks['reflection_y'] = (x+self._Lx*(self._Ly-1-y), 0)
        self._basis = spin_basis_general(self._L, pauli=0, **blocks)
        self.basis = self._basis
        self.dim = self._basis.Ns
        # QuSpin site i is bit N-1-i and full rows are in descending order.
        # Explicit conversion preserves this module's public bit-i convention.
        def reverse_bits(values):
            result = np.zeros_like(values, dtype=np.int64)
            for i in range(self._L):
                result |= ((values >> i) & 1) << (self._L-1-i)
            return result
        representatives = reverse_bits(self._basis.states.astype(np.int64))
        self._basis_masks = representatives
        self.h_ising = self.h_ising[representatives]
        self.h_echo = self.h_echo[representatives]
        self._jz = self._jz[representatives]
        self._quspin_rows = self.full_dim-1-reverse_bits(np.arange(self.full_dim))
        projector = self._basis.get_proj(np.float64)
        self._initial_state = np.asarray(projector.T @ np.full(self.full_dim, 1/np.sqrt(self.full_dim)),
                                         dtype=complex)[:, None]
        self._ground_state = np.asarray(projector.getrow(self.full_dim-1).toarray().T, dtype=complex)
        # Spin inversion preserves spatial reflection sectors. Cache its exact
        # reduced permutation for Dz instead of exponentiating a pi rotation.
        flip = (projector.T @ projector[::-1]).tocsr()
        self._spin_flip = flip
        self.op = {}
        for axis in 'xyz':
            self.op['J'+axis] = hamiltonian(
                [[axis, [[1., i] for i in range(self._L)]]], [], basis=self._basis,
                dtype=np.complex128, check_symm=False, check_herm=False, check_pcon=False).tocsr()

    def to_full(self, state):
        """Expand column states to ascending bit-i computational order."""
        a = self._as_column_state(state)
        if self.backend == 'numpy':
            return a.copy()
        return np.asarray(self._basis.get_proj(np.float64) @ a)[self._quspin_rows]

    def from_full(self, state):
        """Project full column states; reject components outside the selected sector."""
        a = np.asarray(state, complex)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[0] != self.full_dim:
            raise ValueError(f'Full state must have {self.full_dim} rows.')
        if self.backend == 'numpy':
            return a.copy()
        native = np.empty_like(a)
        native[self._quspin_rows] = a
        projector = self._basis.get_proj(np.float64)
        reduced = np.asarray(projector.T @ native)
        if not np.allclose(projector @ reduced, native, atol=1e-12, rtol=1e-10):
            raise ValueError('State is outside the selected reflection sectors; use_reflections=False is required.')
        return reduced

    def _as_column_state(self, state):
        a = np.asarray(state, dtype=complex)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2 or a.shape[0] != self.dim:
            raise ValueError(f'State must have shape ({self.dim},) or ({self.dim}, nvec).')
        return a

    def collective(self, axis, state):
        """Apply J_axis=sum_i sigma_axis,i/2 (sparse in the QuSpin backend)."""
        axis = axis.lower()
        a = self._as_column_state(state)
        if self.backend == 'quspin':
            if axis not in ('x', 'y', 'z'):
                raise ValueError('axis must be x, y or z.')
            return self.op['J'+axis] @ a
        if axis == 'z':
            return self._jz[:, None] * a
        if axis not in ('x', 'y'):
            raise ValueError('axis must be x, y or z.')
        out = np.zeros_like(a)
        for i in range(self._L):
            src = a.reshape(-1, 2, 1 << i, a.shape[1])
            dst = out.reshape(src.shape)
            dst[:, 0] += (0.5 if axis == 'x' else 0.5j) * src[:, 1]
            dst[:, 1] += (0.5 if axis == 'x' else -0.5j) * src[:, 0]
        return out

    def rotate(self, state, axis, angle):
        """Apply exp(-i angle J_axis); return column states."""
        if not np.isfinite(angle):
            raise ValueError('angle must be finite.')
        axis = axis.lower()
        a = self._as_column_state(state).copy()
        if axis == 'z':
            return np.exp(-1j*angle*self._jz)[:, None] * a
        if axis not in ('x', 'y'):
            raise ValueError('axis must be x, y or z.')
        if self.backend == 'quspin':
            generator = (-1j*angle)*self.op['J'+axis]
            return expm_multiply(generator, a, traceA=generator.diagonal().sum())
        c, s = np.cos(angle/2), np.sin(angle/2)
        for i in range(self._L):
            block = a.reshape(-1, 2, 1 << i, a.shape[1])
            lo, hi = block[:, 0].copy(), block[:, 1].copy()
            block[:, 0] = c*lo + (-1j*s if axis == 'x' else s)*hi
            block[:, 1] = c*hi + (-1j*s if axis == 'x' else -s)*lo
        return a

    def evolve(self, state, time, *, echo=False):
        """Echo=True is pure even-z evolution (the spin frame is restored).

        Equivalent to Rx(-pi) U(time/2) Rx(pi) U(time/2).
        'Dz' in run_sequence instead retains the physical final pi rotation.
        """
        if not np.isfinite(time):
            raise ValueError('time must be finite.')
        h = self.h_echo if echo else self.h_ising
        return np.exp(-1j*time*h)[:, None] * self._as_column_state(state)

    def evolve_x(self, state, time, *, echo=False):
        """Many-body Ising evolution with every z-product replaced by x.

        Hx = Ry(pi/2) Hz Ry(-pi/2); echo=True uses the even-order diagonal.
        Fixed basis rotations surround diagonal phase multiplication, retaining
        all cluster contributions without building a dense Hamiltonian. The
        echoed variant restores the spin frame and has no residual pi pulse.
        """
        if not np.isfinite(time):
            raise ValueError('time must be finite.')
        rotated = self.rotate(state, 'y', -np.pi/2)
        evolved = self.evolve(rotated, time, echo=echo)
        return self.rotate(evolved, 'y', np.pi/2)

    def apply_ising_x(self, state, *, echo=False):
        """Apply Hx (or its even-order part) in the working basis."""
        diagonal = self.h_echo if echo else self.h_ising
        rotated = self.rotate(state, 'y', -np.pi/2)
        return self.rotate(diagonal[:, None]*rotated, 'y', np.pi/2)

    def _build_xy_operators(self):
        """Replace only J_ij Zi Zj by J_ij (Xi Xj + Yi Yj).

        J_ij is the Pauli coefficient after summing contributions from every
        occupation cluster through n. All non-pair z terms and the constant
        are retained. XY_echo projects out odd z terms, retaining the same
        exchange. This defines an averaged Hamiltonian, not a single pulse echo.
        """
        if self._xy_operators is not None:
            return
        pairs = [(sites[0], sites[1], value) for sites, value in self.pauli_z.items()
                 if len(sites) == 2 and value != 0]
        masks = self._basis_masks if self.backend == 'quspin' else np.arange(self.dim)
        pair_z = np.zeros(self.dim)
        for i, j, value in pairs:
            pair_z += value*(1 - 2*(((masks >> i) ^ (masks >> j)) & 1))
        if self.backend == 'quspin':
            from quspin.operators import hamiltonian
            # pauli=0 uses spin-1/2 generators: XX+YY = 2(S+S- + S-S+).
            couplings = [[2*value, i, j] for i, j, value in pairs]
            self._xy_exchange = hamiltonian(
                [['+-', couplings], ['-+', couplings]], [], basis=self._basis,
                dtype=np.float64, check_symm=False, check_herm=False, check_pcon=False).tocsr()
            exchange_trace = self._xy_exchange.diagonal().sum()
        else:
            def exchange(state):
                state = np.asarray(state)
                columns = state.reshape(self.dim, -1)
                result = np.zeros_like(columns, dtype=np.result_type(state.dtype, np.float64))
                for i, j, value in pairs:
                    # Swap opposite spins only; no large index/permutation arrays.
                    shape = (-1, 2, 1 << (j-i-1), 2, 1 << i, columns.shape[1])
                    src, dst = columns.reshape(shape), result.reshape(shape)
                    dst[:, 0, :, 1, :, :] += 2*value*src[:, 1, :, 0, :, :]
                    dst[:, 1, :, 0, :, :] += 2*value*src[:, 0, :, 1, :, :]
                return result.reshape(state.shape)
            self._xy_exchange = LinearOperator((self.dim, self.dim), matvec=exchange,
                rmatvec=exchange, matmat=exchange, rmatmat=exchange, dtype=np.float64)
            exchange_trace = 0.
        self._xy_operators, self._xy_traces = {}, {}
        for token, diagonal in (('XY', self.h_ising-pair_z),
                                ('XY_echo', self.h_echo-pair_z)):
            def apply(state, d=diagonal):
                state = np.asarray(state)
                return self._xy_exchange @ state + (d*state if state.ndim == 1 else d[:, None]*state)
            self._xy_operators[token] = LinearOperator((self.dim, self.dim), matvec=apply,
                rmatvec=apply, matmat=apply, rmatmat=apply, dtype=np.float64)
            self._xy_traces[token] = float(exchange_trace+diagonal.sum())

    def xy_operator(self, *, echo=False):
        """Return the cached Hermitian XY LinearOperator in the working basis."""
        self._build_xy_operators()
        return self._xy_operators['XY_echo' if echo else 'XY']

    def evolve_xy(self, state, time, *, echo=False):
        """Apply exp(-i time H_XY), or the parity-filtered H_XY with echo=True.

        Uses exponential multiplication, not diagonal phases or Trotter steps.
        Filtering is not generally equivalent to a finite midpoint-pulse echo.
        """
        if not np.isfinite(time):
            raise ValueError('time must be finite.')
        operator = self.xy_operator(echo=echo)
        token = 'XY_echo' if echo else 'XY'
        return expm_multiply((-1j*time)*operator, self._as_column_state(state),
                             traceA=(-1j*time)*self._xy_traces[token])

    def _validate_sequence(self, theta, gate_sequence):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.ndim != 1 or len(theta) != len(gate_sequence) or not np.all(np.isfinite(theta)):
            raise ValueError('Provide one finite parameter per gate.')
        if any(g not in self.gates for g in gate_sequence):
            raise ValueError(f'Supported gates: {self.gates}')
        return theta

    def _apply_gate(self, state, token, value):
        if token.startswith('R'):
            return self.rotate(state, token[1], value)
        if token in ('Ix', 'Ix_echo'):
            return self.evolve_x(state, value, echo=token == 'Ix_echo')
        if token in ('XY', 'XY_echo'):
            return self.evolve_xy(state, value, echo=token == 'XY_echo')
        state = self.evolve(state, value, echo=token in ('Iz_echo', 'Dz'))
        if token == 'Dz':
            flipped = self._spin_flip @ state if self.backend == 'quspin' else state[::-1].copy()
            state = (-1j)**self._L*flipped
        return state

    def _gate_derivative(self, state, token):
        """Apply -iG; for Dz the fixed pi rotation commutes with H_even."""
        if token.startswith('R'):
            return -1j*self.collective(token[1], state)
        if token in ('Ix', 'Ix_echo'):
            return -1j*self.apply_ising_x(state, echo=token == 'Ix_echo')
        if token in ('XY', 'XY_echo'):
            return -1j*(self.xy_operator(echo=token == 'XY_echo') @ state)
        h = self.h_echo if token in ('Iz_echo', 'Dz') else self.h_ising
        return -1j*h[:, None]*state

    def run_sequence(self, theta, gate_sequence, *, state=None, gradient=False):
        """One parameter per gate; returns psi or (psi, dpsi/dtheta).

        psi has shape (dim,), derivative states have shape (nparam, dim).
        The supplied initial state must be a single normalized vector.
        """
        theta = self._validate_sequence(theta, gate_sequence)
        a = self._as_column_state(self._initial_state if state is None else state).copy()
        if a.shape[1] != 1 or not np.all(np.isfinite(a)) or not np.isclose(np.linalg.norm(a), 1):
            raise ValueError('Initial state must be one finite, normalized vector.')
        for token, value in zip(gate_sequence, theta):
            a = self._apply_gate(a, token, value)
            if gradient:
                derivative = self._gate_derivative(a[:, :1], token)
                a = np.concatenate((a, derivative), axis=1)
        return (a[:, 0], a[:, 1:].T) if gradient else a[:, 0]

    def _run_sequence_second_order(self, theta, gate_sequence):
        """Propagate unique mixed state derivatives, packed by (i,j), i<=j.

        Column layout is [psi, p first derivatives, p*(p+1)/2 second ones].
        This stores one triangle only, without materializing a (p,p,dim) array.
        """
        theta = self._validate_sequence(theta, gate_sequence)
        a = self._initial_state.copy()
        pairs = []
        for k, (token, value) in enumerate(zip(gate_sequence, theta)):
            a = self._apply_gate(a, token, value)
            # New first derivative and mixed derivatives with previous gates.
            new = self._gate_derivative(a[:, :k+1], token)
            diagonal = self._gate_derivative(new[:, :1], token)
            a = np.concatenate((a[:, :k+1], new[:, :1], a[:, k+1:],
                                new[:, 1:], diagonal), axis=1)
            pairs.extend((i, k) for i in range(k+1))
        p = len(theta)
        return a[:, 0], a[:, 1:p+1], a[:, p+1:], pairs

    def squeezing(self, state, state_grad=None):
        """Wineland xi (not xi**2), minimized perpendicular to the mean spin.

        Uses the full centered covariance, including transverse means; valid
        for rotations about any axis and un-echoed odd-z interactions.
        At vanishing mean spin xi=inf (gradient undefined/NaN). At a degenerate
        minimum transverse variance, the returned gradient is one branch.
        """
        psi = self._as_column_state(state)
        if psi.shape[1] != 1 or not np.isclose(np.linalg.norm(psi), 1):
            raise ValueError('Squeezing requires a normalized single state.')
        psi = psi[:, 0]
        jpsi = np.array([self.collective(a, psi)[:, 0] for a in 'xyz'])
        mean = np.real(jpsi @ psi.conj())
        length = np.linalg.norm(mean)
        grad = None if state_grad is None else np.asarray(state_grad, complex)
        if grad is not None and (grad.ndim != 2 or grad.shape[1] != self.dim):
            raise ValueError('state_grad must have shape (nparam, dim).')
        if length < 1e-12:
            return np.inf if grad is None else (np.inf, np.full(len(grad), np.nan))
        covariance = np.real(jpsi.conj() @ jpsi.T) - np.outer(mean, mean)
        normal = mean / length
        ref = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, ref)
        u /= np.linalg.norm(u)
        plane = np.column_stack((u, np.cross(normal, u)))
        vals, vecs = np.linalg.eigh(plane.T @ covariance @ plane)
        variance = max(0., vals[0])
        xi = np.sqrt(self._L*variance)/length
        if grad is None:
            return float(xi)
        direction = plane @ vecs[:, 0]
        dm = 2*np.real(grad.conj() @ jpsi.T)
        jp = direction @ jpsi
        j2p = sum(direction[k]*self.collective(axis, jp)[:, 0] for k, axis in enumerate('xyz'))
        # Envelope derivative includes movement of the transverse plane.
        dv = 2*np.real(grad.conj() @ j2p)
        dv -= 2*(direction @ mean)*(dm @ direction)
        dv -= 2*(direction @ covariance @ normal)/length*(dm @ direction)
        derivative = self._L/(2*xi*length**2)*(dv - 2*variance/length*(dm @ normal)) if xi > 1e-14 else np.full(len(grad), np.nan)
        return float(xi), derivative

    def evaluate_gate_sequence(self, theta, gate_sequence, gradient=False, *, state=None):
        """Return psi, or (psi, dpsi) with gradient=True.

        dpsi[k, b] = d psi[b] / d theta[k], shape (n_gates, self.dim).
        Derivatives use dU/dtheta = -i G U and are propagated through every
        subsequent gate. They are analytic, up to floating-point precision,
        for the chosen cluster-truncated Hamiltonian. The initial state and
        interaction parameters are held fixed.
        """
        return self.run_sequence(theta, gate_sequence, state=state, gradient=gradient)

    def evaluate_gate_sequence_squeezing(self, theta, gate_sequence, gradient=False,
                                        *, return_state=False, hessian=False):
        """Evaluate squeezing and optionally return the evolved state.

        gradient=False: xi, or (xi, psi) when return_state=True.
        gradient=True: (xi, dxi), or (xi, dxi, psi, dpsi) with return_state=True.
        dxi has shape (n_gates,), dpsi has shape (n_gates, self.dim).
        All quantities are obtained from one sequence propagation.
        hessian=True implies gradient=True and returns (xi, dxi, Hxi), or
        (xi, dxi, Hxi, psi, dpsi) with return_state=True. Hxi is (n_gates,n_gates).
        The Hessian is analytic; degenerate transverse variances raise ValueError.
        """
        if hessian:
            psi, first, second, pairs = self._run_sequence_second_order(theta, gate_sequence)
            result = squeezing_second_order(self, psi, first, second, pairs)
            # Copy returned states to release the much larger second-order buffer.
            return (*result, psi.copy(), first.T.copy()) if return_state else result
        result = self.run_sequence(theta, gate_sequence, gradient=gradient)
        if gradient:
            psi, dpsi = result
            xi, dxi = self.squeezing(psi, dpsi)
            return (xi, dxi, psi, dpsi) if return_state else (xi, dxi)
        xi = self.squeezing(result)
        return (xi, result) if return_state else xi

    def evaluate(self, theta, gate_sequence, gradient=False):
        """Short alias for evaluate_gate_sequence_squeezing."""
        return self.evaluate_gate_sequence_squeezing(theta, gate_sequence, gradient)

    def optimize(self, gate_sequence, theta0, *, method=None, bounds=None, options=None,
                 hessian=None, gradient=None, tol=None, callback=None, constraints=(),
                 results_dir=None, study=None):
        """Minimize xi using a selectable SciPy local optimizer.

        method=None preserves L-BFGS-B by default, chooses trust-constr with
        hessian=True, or SLSQP for explicit constraints without an exact Hessian.
        Standard SciPy method names are accepted case-insensitively.

        bounds=None keeps [-pi,pi] for rotations and [0,infinity) for times.
        bounds=False explicitly removes all bounds (required by BFGS, CG,
        Newton-CG, dogleg, trust-exact, trust-ncg and trust-krylov). A list of
        bound pairs or scipy.optimize.Bounds overrides the defaults.

        gradient=None automatically uses analytic derivatives where supported;
        False lets SciPy estimate gradients for methods that permit it.
        hessian=None automatically enables the exact Hessian for Newton-CG,
        dogleg, trust-exact, trust-ncg and trust-krylov. Set hessian=True for
        the exact Hessian with trust-constr; False uses its default approximation.

        tol, callback, constraints and the method-specific options dict pass
        through to scipy.optimize.minimize. Incompatible bounds, constraints,
        or derivative requests raise instead of being silently ignored.
        Returns scipy.optimize.OptimizeResult. See README for examples.
        results_dir enables best-only JSON persistence of successful final
        results. Custom constraints require a stable study label, so workers
        compare solutions to the same constrained problem. Stored records
        contain only parameters and xi; case information is encoded in the path.
        """
        gate_sequence = tuple(gate_sequence)
        theta0 = self._validate_sequence(theta0, gate_sequence)
        if len(theta0) == 0:
            raise ValueError('Optimization requires at least one gate parameter.')
        for name, value in (('gradient', gradient), ('hessian', hessian)):
            if value is not None and not isinstance(value, (bool, np.bool_)):
                raise ValueError(f'{name} must be None, True or False.')
        if constraints is None:
            constraints = ()
        has_constraints = len(constraints) > 0 if isinstance(constraints, (tuple, list, dict)) else True
        methods = ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
                   'TNC', 'COBYLA', 'COBYQA', 'SLSQP', 'dogleg', 'trust-ncg',
                   'trust-exact', 'trust-krylov', 'trust-constr')
        if method is None:
            method = 'trust-constr' if hessian else ('SLSQP' if has_constraints else 'L-BFGS-B')
        names = {m.lower(): m for m in methods}
        if not isinstance(method, str) or method.lower() not in names:
            raise ValueError(f'Unsupported method. Choose from {methods}.')
        method = names[method.lower()]
        bound_methods = {'Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'COBYLA',
                         'COBYQA', 'SLSQP', 'trust-constr'}
        constraint_methods = {'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr'}
        derivative_free = {'Nelder-Mead', 'Powell', 'COBYLA', 'COBYQA'}
        hessian_methods = {'Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact',
                           'trust-krylov', 'trust-constr'}
        required_gradient = {'Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'}
        required_hessian = {'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'}
        use_gradient = method not in derivative_free if gradient is None else bool(gradient)
        use_hessian = method in required_gradient if hessian is None else bool(hessian)
        if use_gradient and method in derivative_free:
            raise ValueError(f'{method} does not use gradients; use gradient=None or False.')
        if not use_gradient and method in required_gradient:
            raise ValueError(f'{method} requires an analytic gradient; use gradient=None or True.')
        if use_hessian and method not in hessian_methods:
            raise ValueError(f'{method} does not use a supplied Hessian; use hessian=False or another method.')
        if not use_hessian and method in required_hessian:
            raise ValueError(f'{method} requires a Hessian; use hessian=None or True.')
        if use_hessian and not use_gradient:
            raise ValueError('The exact Hessian option requires analytic gradients too.')
        if has_constraints and method not in constraint_methods:
            raise ValueError(f'{method} does not support constraints; use SLSQP or trust-constr.')
        if bounds is False:
            bounds = None
        else:
            if method not in bound_methods:
                raise ValueError(f'{method} does not support bounds. Pass bounds=False for an '
                                 'unconstrained run, or choose a method supporting bounds.')
            if bounds is None:
                bounds = [(-np.pi, np.pi) if g.startswith('R') else (0, None) for g in gate_sequence]

        store = None
        if results_dir is not None:
            from .optimization_results import OptimizationResultStore
            if has_constraints and (not isinstance(study, str) or not study.strip()):
                raise ValueError('Saving constrained optimizations requires a study label '
                                 'identifying the constraints, e.g. study="total_time_le_150".')
            store = OptimizationResultStore(results_dir)
            stored_bounds = False if bounds is None else bounds
            result_path = store.path_for(self, gate_sequence, bounds=stored_bounds, study=study)

        # Share objective derivatives across SciPy's separate fun/jac/hess calls.
        # Only the small objective results are cached, not state derivatives.
        cached_theta = cached_result = None
        def objective(theta):
            nonlocal cached_theta, cached_result
            if cached_theta is None or not np.array_equal(theta, cached_theta):
                cached_result = self.evaluate_gate_sequence_squeezing(
                    theta, gate_sequence, gradient=use_gradient, hessian=use_hessian)
                cached_theta = np.array(theta, copy=True)
            return cached_result
        if use_gradient:
            fun = lambda theta: objective(theta)[:2]
        else:
            fun = objective
        kwargs = dict(method=method, jac=True if use_gradient else None,
                      bounds=bounds, constraints=constraints, tol=tol,
                      callback=callback, options=None if options is None else dict(options))
        if use_hessian:
            kwargs['hess'] = lambda theta: objective(theta)[2]
        result = minimize(fun, theta0, **kwargs)
        if store is not None:
            result.saved = False
            result.result_path = str(result_path)
            if result.success and np.isfinite(result.fun) and result.fun >= 0 and np.all(np.isfinite(result.x)):
                result.saved = store.save_if_better(self, gate_sequence, result.x, result.fun,
                                                   bounds=stored_bounds, study=study)
        return result


VarOptMultiBodyIsing = ConstructSquareArray
