"""Process-shared cache of multibody cluster coefficients (no pickle)."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from . import rydberg_cluster


def cached_interactions(directory, lx, ly, n, **kwargs):
    """Reuse exact coefficients across layers/orders, serialized under a lock.

    The key includes every calculation argument and the cluster solver source
    hash. Changes to the solver invalidate previous entries automatically.
    Cache failures are explicit; no partial or incompatible entry is reused.
    """
    identity = json.dumps(dict(version=1, lx=lx, ly=ly, n=n, kwargs=kwargs,
                               solver=hashlib.sha256(Path(rydberg_cluster.__file__).read_bytes()).hexdigest()),
                          sort_keys=True, allow_nan=False)
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'interactions_{key}.npz'
    with path.with_suffix('.npz.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if data['identity'].item() != identity:
                    raise ValueError(f'Interaction cache identity mismatch: {path}')
                result = json.loads(data['metadata'].item())
                result['sites'] = [tuple(site) for site in result['sites']]
                for name in ('occupation', 'ising'):
                    masks, values = data[name+'_masks'], data[name+'_values']
                    if (masks.ndim != 1 or masks.shape != values.shape
                            or not np.all(np.isfinite(values))):
                        raise ValueError(f'Invalid interaction cache: {path}')
                    result[name] = {tuple(i for i in range(lx*ly) if int(mask) >> i & 1): float(v)
                                    for mask, v in zip(masks, values)}
                return result
        result = rydberg_cluster.calculate_interactions(lx, ly, n, **kwargs)
        payload = {'identity': identity,
                   'metadata': json.dumps({k: v for k, v in result.items()
                                           if k not in ('occupation', 'ising')}, allow_nan=False)}
        for name in ('occupation', 'ising'):
            payload[name+'_masks'] = np.asarray(
                [sum(1 << i for i in sites) for sites in result[name]], dtype=np.int64)
            payload[name+'_values'] = np.asarray(list(result[name].values()), dtype=float)
        fd, filename = tempfile.mkstemp(prefix='.interactions_', suffix='.npz', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(filename, path)
        finally:
            Path(filename).unlink(missing_ok=True)
        return result
