"""Locked, versioned numeric cache of reflection-basis geometry operators."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse
import quspin


def cached_geometry(directory, lx, ly, reflections, states, builder, source_file):
    """Cache numeric data only; never deserialize a QuSpin object or pickle.

    Actual basis-state ordering and dependency versions are part of the key.
    Returned buffers belong to each caller, so mutations cannot leak between
    arrays. The physical interaction parameters deliberately do not enter.
    """
    identity = json.dumps(dict(
        version=1, lx=lx, ly=ly, reflections=reflections,
        ordering=hashlib.sha256(np.asarray(states, dtype=np.int64).tobytes()).hexdigest(),
        numpy=np.__version__, scipy=scipy.__version__, quspin=quspin.__version__,
        source=hashlib.sha256(Path(source_file).read_bytes()).hexdigest()), sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'geometry_{key}.npz'
    with path.with_suffix('.npz.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['identity'].item() != identity:
                    raise ValueError(f'Geometry cache identity mismatch: {path}')
                manifest = json.loads(saved['manifest'].item())
                result = {}
                for name, kind in manifest.items():
                    if kind == 'csr':
                        matrix = csr_matrix((saved[name+'_data'], saved[name+'_indices'],
                                             saved[name+'_indptr']), shape=tuple(saved[name+'_shape']))
                        matrix.check_format(full_check=True)
                        result[name] = matrix
                    else:
                        result[name] = saved[name]
                return result
        result = builder()
        payload, manifest = {'identity': identity}, {}
        for name, value in result.items():
            if issparse(value):
                matrix = value.tocsr()
                manifest[name] = 'csr'
                payload.update({name+'_data': matrix.data, name+'_indices': matrix.indices,
                                name+'_indptr': matrix.indptr, name+'_shape': matrix.shape})
            else:
                manifest[name] = 'array'
                payload[name] = value
        payload['manifest'] = json.dumps(manifest, sort_keys=True)
        fd, filename = tempfile.mkstemp(prefix='.geometry_', suffix='.npz', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(filename, path)
        finally:
            Path(filename).unlink(missing_ok=True)
        return result
