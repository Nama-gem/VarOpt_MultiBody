"""Fully echoed multibody ZZ+XX Suzuki-Trotter optimization.

SLURM_ARRAY_TASK_ID=0..499 covers orders 1, 2, 3, 4 and 5 in blocks of 100,
with the same Rb, Omega/Delta and layer grid as cluster_optimization_Ising.
Each layer has one shared time t; Rx rotations appear only between layers.
All orders begin with z evolution. Orders 3/4/5 require interaction-sign reversal.
Absolute times are 2*t (orders 1/2), 41*t/12 (3), (7/(2-2**(1/3))-2)*t (4).
"""

import os
import json
import gc
import ctypes
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize_scalar

from functions import ConstructSquareArray, OptimizationResultStore
import fcntl


# ============================================================
# Settings
# ============================================================

LX = 4
LY = 4

RB_VALUES = [
    1.5,
    2.0,
    2.5,
    3.0,
]

OMEGA_DELTA_RATIOS = [
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 10),
]

TROTTER_ORDERS = (1, 2, 3, 4)
WARM_START_TROTTER_ORDERS = (1, 2, 3, 4)

MAX_LAYERS = 8

N_INITIAL_CONDITIONS = 100
LHS_SEED_BASE = None

# Geometric single-gate scan; reduce GROWTH to resolve narrower minima.
SINGLE_PULSE_SCAN_START = 0.01
SINGLE_PULSE_SCAN_GROWTH = 1.15
SINGLE_PULSE_SCAN_MAX_TIME = 1e6
SINGLE_PULSE_SCAN_TOL = 1e-10
SINGLE_PULSE_CALIBRATION = "suzuki_first_minimum_v1"

# Shared across all layer counts and Trotter orders; None disables disk caching.
INTERACTION_CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache" / "multibody_interactions"


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

method = "L-BFGS-B"
use_hessian = False

# method = "SLSQP"
# use_hessian = False

# method = "trust-constr"
# use_hessian = True

options = {
    "maxiter": 1000,
}

save_best = True


# If True:
#
#     all warm-start jobs
#     + N_INITIAL_CONDITIONS LHS points
#
# are optimized.
#
# If False:
#
#     only warm-start jobs are optimized.
#
# Warm-start parameter vectors are NOT loaded here.
# They are loaded immediately before their optimization begins.
WARM_START_INCLUDE_LHS = False


# If True, rescale only the shared Suzuki-Trotter gate durations.
WARM_START_RESCALE_ISING = True


# ------------------------------------------------------------
# Gradient diagnostic
# ------------------------------------------------------------

# None:
#     Preserve the normal behavior of ConstructSquareArray.optimize().
#
# False:
#     Ask scipy to estimate the gradient numerically.
#
# Normally leave this as None.
OPTIMIZER_GRADIENT = None


# ============================================================
# Silent memory cleanup
# ============================================================

RUN_GC = True
RUN_MALLOC_TRIM = True


def run_garbage_collection():
    if RUN_GC:
        gc.collect()


def run_malloc_trim():
    """Release unused heap on Linux; silently continue elsewhere."""
    if RUN_MALLOC_TRIM:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


# ============================================================
# Warm-start settings
# ============================================================

USE_WARM_START = True


WARM_START_RB_VALUES = [
    1.5,
    2.0,
    2.5,
    3.0,
]


WARM_START_OMEGA_DELTA_RATIOS = [
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 10),
]


# ============================================================
# Helpers
# ============================================================

def format_float_for_path(x):

    if float(x).is_integer():

        return str(
            int(x)
        )

    return str(x).replace(
        ".",
        "p",
    )


def parameter_directory(
    base_dir,
    Lx,
    Ly,
    Rb,
    numerator,
    denominator,
):

    return (
        Path(base_dir)
        / f"{Lx}x{Ly}"
        / f"Rb_{format_float_for_path(Rb)}"
        / f"OmegaDelta_{numerator}_{denominator}"
    )


def project_to_bounds(
    x,
    bounds,
):

    x = np.asarray(
        x,
        dtype=float,
    ).copy()

    bounds = np.asarray(
        bounds,
        dtype=float,
    )

    return np.clip(
        x,
        bounds[:, 0],
        bounds[:, 1],
    )


def save_summary_atomic(
    summary_file,
    *,
    best_fun,
    best_x,
    **metadata,
):
    """
    Save only if best_fun improves on the existing summary.

    Returns True if saved, False otherwise.

    Uses a persistent lock file to protect concurrent writers
    on Linux/macOS. Every writer must use this function.
    """
    best_fun = float(best_fun)

    # Never save a failed/non-finite objective value.
    if not np.isfinite(best_fun):
        return False

    summary_file = Path(summary_file).resolve()
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    lock_file = summary_file.with_name(
        summary_file.name + ".lock"
    )

    # Keep this lock file in place; do not delete it after use.
    with lock_file.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        try:
            # Compare while holding the lock.
            if summary_file.exists():
                with np.load(
                    summary_file,
                    allow_pickle=False,
                ) as saved:
                    previous_best = float(
                        saved["best_fun"].item()
                    )

                if (
                    np.isfinite(previous_best)
                    and best_fun >= previous_best
                ):
                    return False

            # Preserve the original metadata conversions.
            if "last_result_success" in metadata:
                metadata["last_result_success"] = bool(
                    metadata["last_result_success"]
                )

            if "last_result_message" in metadata:
                metadata["last_result_message"] = str(
                    metadata["last_result_message"]
                )

            fd, temporary_name = tempfile.mkstemp(
                prefix=".summary_",
                suffix=".npz",
                dir=summary_file.parent,
            )
            temporary_file = Path(temporary_name)

            try:
                with os.fdopen(fd, "wb") as stream:
                    np.savez(
                        stream,
                        best_fun=best_fun,
                        best_x=np.array(best_x, copy=True),
                        **metadata,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())

                os.replace(temporary_file, summary_file)

            finally:
                temporary_file.unlink(missing_ok=True)

            return True

        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# ============================================================
# Gate sequence
# ============================================================

def gate_sequence(n_layers, ub_Ising, trotter_order=1):
    """One shared interaction time per layer, separated by variational Rx."""
    if trotter_order not in TROTTER_ORDERS:
        raise ValueError("trotter_order must be 1, 2, 3, 4 or 5")
    sequence, bounds = [], []
    for layer in range(n_layers):
        if layer > 0:
            sequence.append("Rx")
            bounds.append((-np.pi, np.pi))
        sequence.append(f"SuzukiTrotter{trotter_order}")
        bounds.append((0, ub_Ising))
    return sequence, bounds


# ============================================================
# Latin hypercube sampling
# ============================================================

def sample_lhs(
    bounds,
    n_samples,
    seed=None,
):

    bounds = np.asarray(
        bounds,
        dtype=float,
    )

    lower = bounds[:, 0]
    upper = bounds[:, 1]

    sampler = qmc.LatinHypercube(
        d=len(bounds),
        seed=seed,
    )

    samples = sampler.random(
        n=n_samples
    )

    return qmc.scale(
        samples,
        lower,
        upper,
    )


# ============================================================
# Single-pulse optimization
# ============================================================

def scan_first_minimum(objective, *, start, growth, max_time, tolerance=1e-10):
    """Bracket the first resolved decreasing-to-increasing turn, then refine.

    This is a finite-resolution scan, not a guarantee against arbitrarily
    narrow unseen minima. Nonfinite values fail rather than bridging a gap.
    No Hessian is needed at the coherent initial state.
    """
    if (not all(np.isfinite(v) for v in (start, growth, max_time, tolerance))
            or not 0 < start < max_time or growth <= 1 or tolerance < 0):
        raise ValueError("Require 0 < start < max_time, growth > 1, tolerance >= 0.")

    def evaluate(t):
        value = float(objective(float(t)))
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite single-gate squeezing at t={t:g}.")
        return value

    times, values = [0.], [evaluate(0.)]
    t = start
    # Keep the lowest sample since a resolved descent; wait for a resolved rise.
    best = None
    while True:
        times.append(t)
        values.append(evaluate(t))
        k = len(times)-1
        if best is None:
            if values[k] < values[k-1]-tolerance:
                best = k
        elif values[k] < values[best]:
            best = k
        elif values[k] > values[best]+tolerance:
            left, middle, right = times[best-1], times[best], t
            result = minimize_scalar(evaluate, bounds=(left, right), method="bounded",
                                     options={"xatol": max(1e-10, middle*1e-9), "maxiter": 500})
            if (not result.success or not left < result.x < right
                    or result.fun > values[best]+tolerance):
                raise RuntimeError("Single-gate minimum refinement failed.")
            return float(result.x), float(result.fun), np.asarray(times), np.asarray(values)
        if t >= max_time:
            break
        t = min(t*growth, max_time)
    raise RuntimeError(
        f"No single-gate minimum bracketed up to t={max_time:g}. "
        "Increase SINGLE_PULSE_SCAN_MAX_TIME or reduce SINGLE_PULSE_SCAN_GROWTH.")


def get_single_pulse_min_time(arr, store, method, use_hessian, options, *, trotter_order):
    """Scan the selected Suzuki gate; solver arguments belong to the main run.

    A separate versioned cache deliberately ignores legacy Iz_echo .npy files.
    Cache identity includes the physical model and scan settings. The lock
    prevents different layer jobs from duplicating the same calibration.
    """
    if trotter_order not in TROTTER_ORDERS:
        raise ValueError("Unsupported Trotter order.")
    gate = f"SuzukiTrotter{trotter_order}"
    store_dir = Path(store.directory)
    store_dir.mkdir(parents=True, exist_ok=True)
    cache_file = store_dir / f"single_pulse_first_minimum_{gate}.npz"
    identity = dict(calibration=SINGLE_PULSE_CALIBRATION, gate=gate,
                    start=SINGLE_PULSE_SCAN_START, growth=SINGLE_PULSE_SCAN_GROWTH,
                    max_time=SINGLE_PULSE_SCAN_MAX_TIME, tolerance=SINGLE_PULSE_SCAN_TOL)
    for key in ('_Lx', '_Ly', 'Rb', 'Omega', 'n', 'convention', 'boundary',
                'backend', 'use_reflections'):
        identity[key] = getattr(arr, key, None)
    identity_json = json.dumps(identity, sort_keys=True)
    with cache_file.with_suffix('.npz.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as saved:
                if str(saved['identity'].item()) == identity_json:
                    time = float(saved['time'].item())
                    if np.isfinite(time) and time > 0:
                        print(f"Loaded first minimum of {gate}: {time:g}")
                        return time
        print(f"Scanning first minimum of {gate}...")
        time, xi, times, values = scan_first_minimum(
            lambda t: arr.evaluate([t], [gate]), start=SINGLE_PULSE_SCAN_START,
            growth=SINGLE_PULSE_SCAN_GROWTH, max_time=SINGLE_PULSE_SCAN_MAX_TIME,
            tolerance=SINGLE_PULSE_SCAN_TOL)
        fd, name = tempfile.mkstemp(prefix='.single_gate_', suffix='.npz', dir=store_dir)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez(stream, identity=identity_json, time=time, xi=xi,
                         scan_times=times, scan_xi=values, trotter_order=trotter_order)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, cache_file)
        finally:
            Path(name).unlink(missing_ok=True)
        print(f"First minimum of {gate}: t={time:g}, xi={xi:g}")
        return time


# ============================================================
# Warm start
# ============================================================

def build_warm_start_jobs(results_root, Rb, numerator, denominator, trotter_order):
    """Schedule immediate grid neighbors, changing exactly one coordinate.

    Adjacency follows the sweep lists, not numerical differences: e.g. 1/5
    neighbors 1/3 and 1/7. Warm-start source lists only filter these neighbors;
    they never bridge over an excluded grid point. The layer count is fixed
    by the caller. Missing files are skipped when each job is executed.
    """
    grids = (RB_VALUES, OMEGA_DELTA_RATIOS, TROTTER_ORDERS)
    rb_matches = [i for i, value in enumerate(RB_VALUES) if np.isclose(value, Rb)]
    if len(rb_matches) != 1:
        raise ValueError("Target Rb must match exactly one sweep-grid point.")
    target = (rb_matches[0], OMEGA_DELTA_RATIOS.index((numerator, denominator)),
              TROTTER_ORDERS.index(trotter_order))
    jobs = []
    for axis, grid in enumerate(grids):
        for offset in (-1, 1):
            index = target[axis] + offset
            if not 0 <= index < len(grid):
                continue
            indices = list(target)
            indices[axis] = index
            source_Rb = RB_VALUES[indices[0]]
            source_num, source_den = OMEGA_DELTA_RATIOS[indices[1]]
            source_order = TROTTER_ORDERS[indices[2]]
            if (not any(np.isclose(source_Rb, value) for value in WARM_START_RB_VALUES)
                    or (source_num, source_den) not in WARM_START_OMEGA_DELTA_RATIOS
                    or source_order not in WARM_START_TROTTER_ORDERS):
                continue
            jobs.append(dict(
                type="warm_start", source_order=source_order, source_Rb=source_Rb,
                source_numerator=source_num, source_denominator=source_den,
                source_ratio_dir=parameter_directory(
                    Path(results_root) / f"order_{source_order}", LX, LY,
                    source_Rb, source_num, source_den)))
    return jobs


# The added angle is in radians; the split preserves the total time.
LAYER_SPLIT_ROTATION_MAX = 0.05
LAYER_SPLIT_FRACTION_RANGE = (0.25, 0.75)


def build_layer_warm_start_jobs(source_ratio_dir, n_layers, trotter_order,
                                Rb, numerator, denominator):
    """Use adjacent depths at the same physical parameters and Trotter order."""
    variants = []
    if n_layers > 1:
        variants.append((n_layers - 1, "split_largest_interaction"))
    if n_layers < MAX_LAYERS:
        variants.extend((n_layers + 1, strategy) for strategy in
                        ("remove_smallest_rotation", "remove_smallest_interaction"))
    return [dict(type="warm_start", source_ratio_dir=Path(source_ratio_dir),
                 source_layers=depth, strategy=strategy, source_order=trotter_order,
                 source_Rb=Rb, source_numerator=numerator, source_denominator=denominator)
            for depth, strategy in variants]


def transform_layer_warm_start(parameters, strategy, rng=None):
    """Transform chronological [t0, angle0, t1, ...] by one layer.

    These are guesses, not identities for finite Suzuki steps. At the first
    interaction, dropping the following Rx leaves only a global phase on
    |+x>. At the last interaction, dropping the preceding Rx preserves the
    rotation-invariant squeezing objective. Interior rotations are combined
    modulo 2*pi. Target time bounds are applied only AFTER the transformation.
    """
    x = np.asarray(parameters, dtype=float)
    if (x.ndim != 1 or x.size == 0 or x.size % 2 != 1
            or not np.all(np.isfinite(x)) or np.any(x[::2] < 0)):
        raise ValueError("Expected finite alternating times/angles with nonnegative times.")
    if strategy == "split_largest_interaction":
        rng = np.random.default_rng() if rng is None else rng
        i = 2 * int(np.argmax(x[::2]))
        fraction = rng.uniform(*LAYER_SPLIT_FRACTION_RANGE)
        angle = rng.uniform(-LAYER_SPLIT_ROTATION_MAX, LAYER_SPLIT_ROTATION_MAX)
        return np.concatenate((x[:i], [fraction*x[i], angle, (1-fraction)*x[i]], x[i+1:]))
    if x.size < 3:
        raise ValueError("Cannot shorten a single-layer sequence.")
    if strategy == "remove_smallest_rotation":
        angles = (x[1::2] + np.pi) % (2*np.pi) - np.pi
        i = 2 * int(np.argmin(np.abs(angles))) + 1
        return np.concatenate((x[:i-1], [x[i-1]+x[i+1]], x[i+2:]))
    if strategy == "remove_smallest_interaction":
        i = 2 * int(np.argmin(x[::2]))
        if i == 0:
            return x[2:].copy()
        if i == x.size - 1:
            return x[:-2].copy()
        angle = (x[i-1] + x[i+1] + np.pi) % (2*np.pi) - np.pi
        return np.concatenate((x[:i-1], [angle], x[i+2:]))
    raise ValueError(f"Unknown layer warm-start strategy: {strategy}")


def load_layer_warm_start(job, target_bounds, target_single_pulse_min_time, rng=None):
    """Read the latest source summary, transform, then enforce target bounds."""
    sequence, source_bounds = gate_sequence(job["source_layers"], np.inf, job["source_order"])
    x = load_warm_start(job["source_ratio_dir"], job["source_layers"], sequence,
                        source_bounds, target_single_pulse_min_time, rescale_ising=False)
    x = transform_layer_warm_start(x, job["strategy"], rng)
    if len(x) != len(target_bounds):
        raise ValueError("Layer transformation does not match target depth.")
    bounded = project_to_bounds(x, target_bounds)
    print(f"Layer warm start: {job['strategy']} from {job['source_layers']} layers")
    if not np.array_equal(x, bounded):
        print("Transformed parameters clipped to target bounds.")
    print("Transformed starting parameters:", bounded)
    return bounded


def load_warm_start(
    source_ratio_dir,
    n_layers,
    target_sequence,
    target_bounds,
    target_single_pulse_min_time,
    rescale_ising=True,
):
    """
    Load an optimum from another (Rb, Omega/Delta) realization.

    IMPORTANT:
        This function is called immediately before the
        corresponding optimization begins.

        Therefore, if another Slurm job has improved its
        summary.npz since this task started, this job uses the
        newest available parameters.

    Shared Suzuki-Trotter times are rescaled according to

        target_single_pulse_min_time
        ---------------------------------
        source_single_pulse_min_time

    Rx parameters are copied WITHOUT rescaling.
    """

    source_layer_dir = (
        Path(source_ratio_dir)
        / f"layers_{n_layers}"
    )

    summary_file = (
        source_layer_dir
        / "summary.npz"
    )

    if not summary_file.exists():

        raise FileNotFoundError(
            "\nWarm-start summary does not exist:\n"
            f"    {summary_file.resolve()}\n"
        )

    print()
    print("=" * 60)
    print("LOADING WARM START")
    print("=" * 60)

    print(
        "Source summary:"
    )

    print(
        summary_file.resolve()
    )

    with np.load(
        summary_file, allow_pickle=False
    ) as data:

        x0 = np.asarray(
            data["best_x"],
            dtype=float,
        )

        source_single_pulse_min_time = float(
            data[
                "single_pulse_min_time"
            ]
        )

        source_calibration = (str(data["single_pulse_calibration"].item())
                              if "single_pulse_calibration" in data else "legacy_Ising")

        source_Rb = float(
            data["Rb"]
        )

        source_num = int(
            data["numerator"]
        )

        source_den = int(
            data["denominator"]
        )

        source_layers = int(
            data["n_layers"]
        )

        source_best_fun = float(
            data["best_fun"]
        )


    print()
    print(
        f"Source Rb              : "
        f"{source_Rb}"
    )

    print(
        f"Source Omega / Delta   : "
        f"{source_num}/{source_den}"
    )

    print(
        f"Source layers          : "
        f"{source_layers}"
    )

    print(
        f"Source objective       : "
        f"{source_best_fun}"
    )

    print(
        f"Source single-pulse t  : "
        f"{source_single_pulse_min_time}"
    )

    print(
        f"Target single-pulse t  : "
        f"{target_single_pulse_min_time}"
    )


    # --------------------------------------------------------
    # Check parameter count
    # --------------------------------------------------------

    if source_layers != n_layers:
        raise ValueError("Warm-start layer count does not match target.")
    if x0.ndim != 1 or not np.all(np.isfinite(x0)):
        raise ValueError("Warm-start parameters must be a finite vector.")

    if len(x0) != len(
        target_sequence
    ):

        raise ValueError(
            "\nWarm-start parameter count does not "
            "match target sequence.\n"
            f"Source parameters : {len(x0)}\n"
            f"Target parameters : "
            f"{len(target_sequence)}\n"
        )


    # --------------------------------------------------------
    # Rescale only Ising interaction times
    # --------------------------------------------------------

    if rescale_ising and source_calibration != SINGLE_PULSE_CALIBRATION:
        print("Legacy/different warm-start calibration: copying shared times without rescaling.")
        rescale_ising = False

    if rescale_ising:

        if (not np.isfinite(source_single_pulse_min_time)
                or source_single_pulse_min_time <= 0
                or not np.isfinite(target_single_pulse_min_time)
                or target_single_pulse_min_time <= 0):
            raise ValueError("Warm-start time scales must be positive and finite.")

        scale = (
            target_single_pulse_min_time
            / source_single_pulse_min_time
        )

        print()
        print(
            f"Rescaling Ising times by: "
            f"{scale}"
        )

        for i, gate in enumerate(
            target_sequence
        ):

            if gate in (
                "SuzukiTrotter1",
                "SuzukiTrotter2",
                "SuzukiTrotter3",
                "SuzukiTrotter4",
                "SuzukiTrotter5",
            ):

                x0[i] *= scale

            elif gate == "Rx":

                # Explicitly leave Rx unchanged.
                pass

    else:

        print()
        print(
            "Ising-time rescaling disabled."
        )


    # --------------------------------------------------------
    # Project onto target bounds
    # --------------------------------------------------------

    x0_before_projection = (
        x0.copy()
    )

    x0 = project_to_bounds(
        x0,
        target_bounds,
    )

    if not np.allclose(
        x0,
        x0_before_projection,
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Warm-start parameters were "
            "projected onto target bounds."
        )


    # --------------------------------------------------------
    # Print resulting warm start
    # --------------------------------------------------------

    print()
    print(
        "Warm-start parameters:"
    )

    for i, (
        gate,
        parameter,
    ) in enumerate(
        zip(
            target_sequence,
            x0,
        )
    ):

        print(
            f"{i:3d}  "
            f"{gate:10s}  "
            f"{parameter:.12g}"
        )

    print("=" * 60)

    return x0


# ============================================================
# Decode Slurm task
# ============================================================

def decode_task_id(task_id):
    tasks_per_order = len(RB_VALUES)*len(OMEGA_DELTA_RATIOS)*MAX_LAYERS
    n_tasks = len(TROTTER_ORDERS)*tasks_per_order
    if not 0 <= task_id < n_tasks:
        raise ValueError(f"task_id must be between 0 and {n_tasks-1}, got {task_id}")
    order_index, remainder = divmod(task_id, tasks_per_order)
    rb_index, remainder = divmod(remainder, len(OMEGA_DELTA_RATIOS)*MAX_LAYERS)
    ratio_index, layer_index = divmod(remainder, MAX_LAYERS)
    numerator, denominator = OMEGA_DELTA_RATIOS[ratio_index]
    return (RB_VALUES[rb_index], numerator, denominator, numerator/denominator,
            layer_index+1, TROTTER_ORDERS[order_index])


# ============================================================
# Main
# ============================================================

def main():

    task_id = int(
        os.environ.get(
            "SLURM_ARRAY_TASK_ID",
            0,
        )
    )

    (
        Rb,
        numerator,
        denominator,
        omega_over_delta,
        n_layers,
        trotter_order,
    ) = decode_task_id(
        task_id
    )


    # ========================================================
    # Task information
    # ========================================================

    print("=" * 60)

    print(
        f"Task ID         : "
        f"{task_id}"
    )

    print(
        f"Geometry        : "
        f"{LX}x{LY}"
    )

    print(
        f"Rb              : "
        f"{Rb}"
    )

    print(
        f"Omega / Delta   : "
        f"{numerator}/{denominator} "
        f"= {omega_over_delta}"
    )

    print(
        f"Number layers   : "
        f"{n_layers}"
    )

    print(
        f"Warm start      : "
        f"{USE_WARM_START}"
    )

    print(
        f"Optimizer       : "
        f"{method}"
    )

    print(
        f"Gradient setting: "
        f"{OPTIMIZER_GRADIENT}"
    )



    print("=" * 60)

    print(f"Suzuki-Trotter order: {trotter_order}")



    # ========================================================
    # Physical parameters
    # ========================================================

    Delta = 1.0

    Omega = (
        omega_over_delta
        * Delta
    )

    print(
        f"Delta           : "
        f"{Delta}"
    )

    print(
        f"Omega           : "
        f"{Omega}"
    )


    # ========================================================
    # Construct array
    # ========================================================


    arr = ConstructSquareArray(
        Lx=LX,
        Ly=LY,
        Rb=Rb,
        Omega=Omega,
        n=12,
        boundary="open",
        backend="quspin",
        use_reflections=True,
        interaction_cache_dir=INTERACTION_CACHE_DIR,
    )



    # ========================================================
    # Results directories
    # ========================================================

    results_base_dir = Path(
        "results_SuzukiTrotter"
    )

    results_base_dir = results_base_dir / f"order_{trotter_order}"

    ratio_dir = (
        parameter_directory(
            base_dir=results_base_dir,
            Lx=LX,
            Ly=LY,
            Rb=Rb,
            numerator=numerator,
            denominator=denominator,
        )
    )

    layer_dir = (
        ratio_dir
        / f"layers_{n_layers}"
    )

    layer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Results directory:"
    )

    print(
        layer_dir.resolve()
    )


    # ========================================================
    # Result stores
    # ========================================================

    single_pulse_store = (
        OptimizationResultStore(
            directory=ratio_dir
        )
    )

    store = OptimizationResultStore(
        directory=layer_dir
    )


    # ========================================================
    # Find/load single-pulse minimum
    # ========================================================

    single_pulse_min_time = (
        get_single_pulse_min_time(
            arr=arr,
            store=single_pulse_store,
            trotter_order=trotter_order,
            method=method,
            use_hessian=use_hessian,
            options=options,
        )
    )


    # ========================================================
    # Ising interaction upper bound
    # ========================================================

    ub_Ising = (
        1.5
        * single_pulse_min_time
    )

    print(
        f"Upper Ising bound: "
        f"{ub_Ising}"
    )


    # ========================================================
    # Build sequence
    # ========================================================

    sequence, bounds = (
        gate_sequence(
            n_layers=n_layers,
            ub_Ising=ub_Ising,
            trotter_order=trotter_order,
        )
    )

    print()
    print(
        "Sequence:"
    )

    print(
        sequence
    )

    print()
    print(
        "Bounds:"
    )

    print(
        bounds
    )


    # ========================================================
    # Build optimization-job list
    # ========================================================
    #
    # Warm-start PARAMETERS are deliberately NOT loaded here.
    #
    # Only the source directories are stored.
    #
    # Each source summary.npz is loaded immediately before its
    # corresponding optimization begins.
    # ========================================================

    layer_rng = np.random.default_rng(
        None if LHS_SEED_BASE is None else LHS_SEED_BASE + task_id)
    optimization_jobs = []


    # --------------------------------------------------------
    # Warm-start jobs
    # --------------------------------------------------------

    if USE_WARM_START:
        optimization_jobs.extend(build_warm_start_jobs(
            results_base_dir.parent, Rb, numerator, denominator, trotter_order))
        optimization_jobs.extend(build_layer_warm_start_jobs(
            ratio_dir, n_layers, trotter_order, Rb, numerator, denominator))


    # --------------------------------------------------------
    # Latin-hypercube jobs
    # --------------------------------------------------------

    use_lhs = (
        not USE_WARM_START
        or WARM_START_INCLUDE_LHS
    )

    if use_lhs:

        if LHS_SEED_BASE is None:

            seed = None

        else:

            seed = (
                LHS_SEED_BASE
                + task_id
            )

        lhs_conditions = (
            sample_lhs(
                bounds=bounds,
                n_samples=N_INITIAL_CONDITIONS,
                seed=seed,
            )
        )

        for x0 in lhs_conditions:

            optimization_jobs.append(
                {
                    "type": "lhs",
                    "x0": np.asarray(
                        x0,
                        dtype=float,
                    ),
                }
            )


    if len(
        optimization_jobs
    ) == 0:

        raise RuntimeError(
            "No optimization jobs were generated."
        )


    # ========================================================
    # Optimization-job summary
    # ========================================================

    n_scheduled_warm_starts = sum(
        job["type"] == "warm_start"
        for job in optimization_jobs
    )

    n_scheduled_lhs = sum(
        job["type"] == "lhs"
        for job in optimization_jobs
    )

    print()
    print("=" * 60)
    print("OPTIMIZATION JOBS")
    print("=" * 60)

    print(
        "Total scheduled jobs:",
        len(optimization_jobs),
    )

    print(
        "Warm-start jobs:",
        n_scheduled_warm_starts,
    )

    print(
        "LHS jobs:",
        n_scheduled_lhs,
    )

    if USE_WARM_START:

        print(
            "Warm-start Rb values:",
            WARM_START_RB_VALUES,
        )

        print(
            "Warm-start Omega/Delta ratios:",
            WARM_START_OMEGA_DELTA_RATIOS,
        )

        print(
            "Rescale Ising times:",
            WARM_START_RESCALE_ISING,
        )

        print()
        print(
            "Warm-start summary files will be "
            "loaded immediately before each "
            "optimization."
        )



    # ========================================================
    # Run optimizations
    # ========================================================

    best_fun = np.inf
    best_x = None

    summary_file = (
        layer_dir
        / "summary.npz"
    )

    n_jobs = len(
        optimization_jobs
    )

    completed_optimizations = 0


    for job_index, job in enumerate(
        optimization_jobs
    ):

        print()
        print("=" * 60)

        print(
            f"Optimization job "
            f"{job_index + 1}/"
            f"{n_jobs}"
        )

        print(
            f"Job type: "
            f"{job['type'].upper()}"
        )

        print("=" * 60)


        # ====================================================
        # Obtain initial condition
        # ====================================================

        if job["type"] == "warm_start":

            print(f"Warm-start source order: {job['source_order']}")

            source_Rb = (
                job["source_Rb"]
            )

            source_numerator = (
                job["source_numerator"]
            )

            source_denominator = (
                job["source_denominator"]
            )

            source_ratio_dir = (
                job["source_ratio_dir"]
            )

            source_layers = job.get("source_layers", n_layers)
            source_summary_file = (
                source_ratio_dir
                / f"layers_{source_layers}"
                / "summary.npz"
            )

            print()
            print(
                "About to load warm start from:"
            )

            print(
                source_summary_file.resolve()
            )


            # -----------------------------------------------
            # Check existence NOW
            # -----------------------------------------------

            if not source_summary_file.exists():

                print()
                print(
                    "Warm-start summary is not available "
                    "at optimization time."
                )

                print(
                    "Skipping this warm-start job:"
                )

                print(
                    f"Rb = {source_Rb}, "
                    f"Omega/Delta = "
                    f"{source_numerator}/"
                    f"{source_denominator}"
                )

                print(
                    "Missing file:"
                )

                print(
                    source_summary_file.resolve()
                )

                continue


            # -----------------------------------------------
            # Load newest summary immediately before optimize
            # -----------------------------------------------

            try:

                if "strategy" in job:
                    x0 = load_layer_warm_start(job, bounds, single_pulse_min_time, layer_rng)
                else:
                    x0 = load_warm_start(
                        source_ratio_dir=source_ratio_dir,
                        n_layers=n_layers,
                        target_sequence=sequence,
                        target_bounds=bounds,
                        target_single_pulse_min_time=(
                            single_pulse_min_time
                        ),
                        rescale_ising=(
                            WARM_START_RESCALE_ISING
                        ),
                    )

            except FileNotFoundError:

                print()
                print(
                    "Warm-start summary disappeared before "
                    "it could be loaded. Skipping job."
                )

                continue


        elif job["type"] == "lhs":

            x0 = np.asarray(
                job["x0"],
                dtype=float,
            ).copy()


        else:

            raise ValueError(
                f"Unknown optimization job type: "
                f"{job['type']}"
            )


        # ====================================================
        # Memory before
        # ====================================================



        # ====================================================
        # Optimization
        # ====================================================

        result = arr.optimize(
            sequence,
            x0,
            method=method,
            hessian=use_hessian,
            gradient=OPTIMIZER_GRADIENT,
            bounds=bounds,
            options=options,
            results_dir=(
                store.directory
                if save_best
                else None
            ),
        )


        # ====================================================
        # Memory immediately after
        # ====================================================



        # ====================================================
        # Extract required result information
        # ====================================================

        current_fun = float(
            result.fun
        )

        current_x = np.array(
            result.x,
            dtype=float,
            copy=True,
        )

        completed_optimizations += 1

        print(
            f"Objective: {current_fun}"
        )

        print(
            f"Success:   {result.success}"
        )

        print(
            f"Message:   {result.message}"
        )

        if hasattr(
            result,
            "nfev",
        ):

            print(
                f"nfev:      {result.nfev}"
            )

        if hasattr(
            result,
            "njev",
        ):

            print(
                f"njev:      {result.njev}"
            )

        if hasattr(
            result,
            "nit",
        ):

            print(
                f"nit:       {result.nit}"
            )


        # ====================================================
        # Update best result
        # ====================================================
        #
        # Accept every completed optimization returning a
        # finite objective, even if scipy sets success=False.
        # ====================================================

        if np.isfinite(
            current_fun
        ):

            if (
                best_x is None
                or current_fun < best_fun
            ):

                best_fun = (
                    current_fun
                )

                best_x = (
                    current_x.copy()
                )

                print(
                    "New best result:",
                    best_fun,
                )

        else:

            print()
            print(
                "WARNING: optimization returned a "
                "non-finite objective."
            )


        # ====================================================
        # Save summary after EVERY completed optimization
        # ====================================================

        if best_x is not None:

            save_summary_atomic(
                summary_file,
                task_id=task_id,
                Lx=LX,
                Ly=LY,
                Rb=Rb,
                numerator=numerator,
                denominator=denominator,
                omega_over_delta=(
                    omega_over_delta
                ),
                n_layers=n_layers,
                trotter_order=trotter_order,
                single_pulse_calibration=SINGLE_PULSE_CALIBRATION,
                gate_sequence=np.asarray(sequence),
                best_fun=best_fun,
                best_x=best_x,
                single_pulse_min_time=(
                    single_pulse_min_time
                ),
                ub_Ising=ub_Ising,
                completed_optimizations=(
                    completed_optimizations
                ),
                attempted_jobs=(
                    job_index + 1
                ),
                total_jobs=n_jobs,
                last_result_success=(
                    result.success
                ),
                last_result_message=(
                    result.message
                ),
            )

            print()
            print(
                "Summary updated:"
            )

            print(
                summary_file.resolve()
            )

            print(
                f"Best objective so far: "
                f"{best_fun}"
            )

            print(
                f"Completed optimizations: "
                f"{completed_optimizations}"
            )


        # ====================================================
        # Explicitly release objects
        # ====================================================

        del result
        del current_x
        del x0


        # ====================================================
        # Garbage collection
        # ====================================================

        run_garbage_collection()



        # ====================================================
        # glibc malloc_trim
        # ====================================================

        run_malloc_trim()


        print(
            "-" * 60
        )


    # ========================================================
    # Check that at least one optimization completed
    # ========================================================

    if best_x is None:

        raise RuntimeError(
            "No usable optimization result was obtained."
        )


    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 60)
    print("FINISHED")
    print("=" * 60)

    print(
        f"Geometry       : "
        f"{LX}x{LY}"
    )

    print(
        f"Rb             : "
        f"{Rb}"
    )

    print(
        f"Omega / Delta  : "
        f"{numerator}/{denominator}"
    )

    print(
        f"Layers         : "
        f"{n_layers}"
    )

    print(
        f"Gate sequence  : "
        f"{sequence}"
    )

    print(
        f"Single Ising t : "
        f"{single_pulse_min_time}"
    )

    print(
        f"Best objective : "
        f"{best_fun}"
    )

    print(
        "Best parameters:"
    )

    print(
        best_x
    )

    print(
        f"Completed opts : "
        f"{completed_optimizations}"
    )

    print(
        f"Summary saved  : "
        f"{summary_file.resolve()}"
    )



if __name__ == "__main__":
    main()