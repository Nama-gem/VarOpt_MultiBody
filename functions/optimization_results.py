"""Best-only optimization records for cooperating macOS/Linux worker processes.

Each JSON file contains exactly {"parameters": [...], "xi": ...}. A stable
sidecar flock protects the whole read/compare/replace transaction. Atomic
replacement ensures readers never see half-written JSON. Locks are released
by the OS if a worker exits; do not delete lock files while workers are active.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import numpy as np


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / 'data' / 'optimization_results'
# Bump if physics, gate definitions, or the objective convention changes.
MODEL_VERSION = 'multibody-xy-pair-replacement-xi-v1'


def _bounds_key(gates, bounds):
    if bounds is False:
        low, high = np.full(len(gates), -np.inf), np.full(len(gates), np.inf)
    elif bounds is None:
        low = np.array([-np.pi if g.startswith('R') else 0. for g in gates])
        high = np.array([np.pi if g.startswith('R') else np.inf for g in gates])
    elif hasattr(bounds, 'lb') and hasattr(bounds, 'ub'):
        low = np.broadcast_to(np.asarray(bounds.lb, float), (len(gates),))
        high = np.broadcast_to(np.asarray(bounds.ub, float), (len(gates),))
    else:
        pairs = list(bounds)
        if len(pairs) != len(gates) or any(len(pair) != 2 for pair in pairs):
            raise ValueError('There must be one lower/upper bound pair per gate.')
        low = np.array([-np.inf if pair[0] is None else pair[0] for pair in pairs], float)
        high = np.array([np.inf if pair[1] is None else pair[1] for pair in pairs], float)
    if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
        raise ValueError('Invalid parameter bounds.')
    return [[float(a).hex(), float(b).hex()] for a, b in zip(low, high)]


def _slug(value, limit=64):
    return re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(value)).strip('.-')[:limit] or 'default'


class OptimizationResultStore:
    """One best-result JSON per system, sequence, bounds and optional study.

    ``directory=None`` uses data/optimization_results in the project root.
    Pass the same shared directory to every worker. Cooperating processes must
    use this API and a filesystem supporting flock and atomic same-directory
    rename (normal local filesystems; check shared-cluster filesystem support).
    File synchronization services between separate machines are not locks.
    """
    def __init__(self, directory=None, *, lock_timeout=30.):
        self.directory = (DEFAULT_RESULTS_DIR if directory is None else Path(directory).expanduser()).resolve()
        if not math.isfinite(lock_timeout) or lock_timeout < 0:
            raise ValueError('lock_timeout must be finite and nonnegative.')
        self.lock_timeout = float(lock_timeout)

    def path_for(self, arr, gate_sequence, *, bounds=None, study=None):
        """
        Deterministic path, without creating files or directories.

        The provided self.directory is used directly. System parameters,
        physical parameters, and study labels are included in the hash,
        but are not added as extra subdirectories.
        """

        gates = tuple(gate_sequence)

        if not gates or any(g not in arr.gates for g in gates):
            raise ValueError(
                'A nonempty sequence of supported gates is required.'
            )

        if study is not None and (
                not isinstance(study, str) or not study.strip()
        ):
            raise ValueError(
                'study must be None or a nonempty string.'
            )

        state = np.ascontiguousarray(
            arr._initial_state,
            dtype='<c16',
        )

        case = dict(
            model=MODEL_VERSION,
            Lx=arr._Lx,
            Ly=arr._Ly,
            n=arr.n,
            Rb=float(arr.Rb).hex(),
            Omega=float(arr.Omega).hex(),
            boundary=arr.boundary,
            convention=arr.convention,
            backend=arr.backend,
            reflections=arr.use_reflections,
            initial_state=hashlib.sha256(
                state.tobytes()
            ).hexdigest(),
            gates=gates,
            bounds=_bounds_key(gates, bounds),
            study=study,
        )

        digest = hashlib.sha256(
            json.dumps(
                case,
                sort_keys=True,
                separators=(',', ':'),
            ).encode()
        ).hexdigest()

        filename = (
            f'{_slug("_".join(gates))}'
            f'--{digest}.json'
        )

        return self.directory / filename

    @contextmanager
    def _lock(self, path, *, exclusive):
        # Never lock the result file itself: os.replace changes its inode.
        with Path(str(path)+'.lock').open('a+b') as handle:
            mode = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
            deadline = time.monotonic()+self.lock_timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), mode)
                    break
                except BlockingIOError:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f'Timed out waiting for optimization result lock: {path}')
                    time.sleep(min(.05, remaining))
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _record(parameters, xi, size):
        if isinstance(xi, (bool, np.bool_)) or not np.isscalar(xi) or np.iscomplexobj(xi):
            raise ValueError('xi must be a finite nonnegative real scalar.')
        xi = float(xi)
        if not math.isfinite(xi) or xi < 0:
            raise ValueError('xi must be a finite nonnegative real scalar.')
        if np.iscomplexobj(parameters):
            raise ValueError('Gate parameters must be real.')
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (size,) or not np.all(np.isfinite(parameters)):
            raise ValueError('Provide one finite parameter per gate.')
        return dict(parameters=parameters.tolist(), xi=xi)

    def _read(self, path, size):
        try:
            with path.open('r', encoding='utf-8') as handle:
                value = json.load(handle)
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f'Invalid optimization record; refusing to overwrite: {path}') from exc
        try:
            if not isinstance(value, dict) or set(value) != {'parameters', 'xi'}:
                raise ValueError('Unexpected fields.')
            return self._record(value['parameters'], value['xi'], size)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid optimization record; refusing to overwrite: {path}') from exc

    def load_best(self, arr, gate_sequence, *, bounds=None, study=None):
        """Return {parameters, xi}, or None when no result exists.

        This is a snapshot. save_if_better always re-reads under its own lock,
        so callers need not perform a separate read before submitting a result.
        Corruption, lock timeouts and I/O failures raise rather than looking
        like a missing optimum. A missing record does not create data files.
        """
        gates = tuple(gate_sequence)
        path = self.path_for(arr, gates, bounds=bounds, study=study)
        if not path.exists():
            return None
        with self._lock(path, exclusive=False):
            return self._read(path, len(gates))

    def save_if_better(self, arr, gate_sequence, parameters, xi, *, bounds=None, study=None):
        """Atomically insert or strictly improve a best record. Return bool.

        Equal/worse xi leaves the JSON bytes and modification time untouched.
        Values must already represent a valid run of this system and study;
        this low-level method does not recompute xi or validate custom constraints.
        """
        gates = tuple(gate_sequence)
        candidate = self._record(parameters, xi, len(gates))
        path = self.path_for(arr, gates, bounds=bounds, study=study)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(path, exclusive=True):
            current = self._read(path, len(gates))
            if current is not None and candidate['xi'] >= current['xi']:
                return False
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                                 prefix='.'+path.stem[:24]+'-', suffix='.tmp',
                                                 delete=False) as handle:
                    temp_path = Path(handle.name)
                    json.dump(candidate, handle, allow_nan=False, separators=(',', ':'))
                    handle.write('\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
                # Persist the directory entry as well as the new file contents.
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
            return True
