# cluster_optimization_XY.py

import os
import gc
import ctypes
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from functions import ConstructSquareArray, OptimizationResultStore


# ============================================================
# Settings
# ============================================================

LX = 4
LY = 4

RB_VALUES = [
    1.5,
    2.0,
    2.5,
]

OMEGA_DELTA_RATIOS = [
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 10),
]

MAX_LAYERS = 5

N_INITIAL_CONDITIONS = 1000
LHS_SEED_BASE = None


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
OPTIMIZER_GRADIENT = False


# ============================================================
# Memory diagnostics
# ============================================================

MEMORY_DIAGNOSTICS = True

RUN_GC = True

# On Linux/Terra, ask glibc to return unused heap memory
# to the operating system after each optimization.
RUN_MALLOC_TRIM = True


def get_linux_memory():
    """
    Return Linux process memory information in MB.

    VmRSS:
        Current resident memory.

    VmHWM:
        Peak resident memory ("high water mark").

    VmSize:
        Current virtual memory size.

    Returns NaN values on systems without /proc/self/status.
    """

    result = {
        "VmRSS": np.nan,
        "VmHWM": np.nan,
        "VmSize": np.nan,
    }

    status_file = Path("/proc/self/status")

    if not status_file.exists():
        return result

    with status_file.open() as f:
        for line in f:
            for key in result:
                if line.startswith(key + ":"):
                    value_kb = float(line.split()[1])
                    result[key] = value_kb / 1024.0

    return result


def print_memory(label):
    """
    Print current and peak process memory.
    """

    if not MEMORY_DIAGNOSTICS:
        return

    memory = get_linux_memory()

    print()
    print(f"[MEMORY] {label}")
    print(f"    VmRSS  = {memory['VmRSS']:.1f} MB")
    print(f"    VmHWM  = {memory['VmHWM']:.1f} MB")
    print(f"    VmSize = {memory['VmSize']:.1f} MB")


def run_garbage_collection():
    """
    Run Python garbage collection.
    """

    if not RUN_GC:
        return

    n_collected = gc.collect()

    if MEMORY_DIAGNOSTICS:
        print(
            f"[MEMORY] gc.collect() collected "
            f"{n_collected} objects"
        )


def run_malloc_trim():
    """
    Ask glibc to return unused heap memory to the OS.

    Useful on Linux/Terra. Silently continues if unavailable.
    """

    if not RUN_MALLOC_TRIM:
        return

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)

        if MEMORY_DIAGNOSTICS:
            print("[MEMORY] malloc_trim(0) called")

    except Exception as exc:
        if MEMORY_DIAGNOSTICS:
            print(
                "[MEMORY] malloc_trim unavailable:",
                repr(exc),
            )


# ============================================================
# Warm-start settings
# ============================================================

USE_WARM_START = True


WARM_START_RB_VALUES = [
    1.5,
    2.0,
    2.5,
]


WARM_START_OMEGA_DELTA_RATIOS = [
    (1, 2),
    (1, 3),
    (1, 5),
    (1, 7),
    (1, 10),
]


# If True:
#
#     all available warm starts
#     + N_INITIAL_CONDITIONS LHS points
#
# are optimized.
#
# If False:
#
#     only available warm starts are optimized.
WARM_START_INCLUDE_LHS = True


# Rescale XY_echo durations using the ratio of single-pulse
# first-minimum times.
#
# Rx rotations are NOT rescaled.
WARM_START_RESCALE_XY = True


# ============================================================
# Helpers
# ============================================================

def format_float_for_path(x):

    if float(x).is_integer():
        return str(int(x))

    return str(x).replace(".", "p")


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


# ============================================================
# Gate sequence
# ============================================================

def gate_sequence(
    n_layers,
    ub_XY,
):
    """
    Construct

        L = 1:
            XY_echo

        L = 2:
            XY_echo, Rx, XY_echo

        L = 3:
            XY_echo, Rx, XY_echo, Rx, XY_echo

        ...

    The number of XY_echo gates is therefore n_layers.

    Rx:
        angle in [-pi, pi]

    XY_echo:
        time in [0, ub_XY]
    """

    sequence = []
    bounds = []

    for layer in range(n_layers):

        if layer > 0:

            sequence.append("Rx")
            bounds.append(
                (-np.pi, np.pi)
            )

        sequence.append("XY_echo")
        bounds.append(
            (0, ub_XY)
        )

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
# Single-pulse XY_echo optimization
# ============================================================

def get_single_pulse_min_time(
    arr,
    store,
    method,
    use_hessian,
    options,
):
    """
    Find/load the time of the first minimum obtained for a
    single XY_echo pulse.

    This provides the characteristic interaction time used for:

        1. defining the upper bound of all XY_echo gates;
        2. rescaling XY_echo durations when loading warm starts.

    The value is cached separately for each Rb and Omega/Delta
    parameter point.
    """

    store_dir = Path(
        store.directory
    )

    store_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        store_dir
        / "single_pulse_min_time.npy"
    )

    # --------------------------------------------------------
    # Load cached result
    # --------------------------------------------------------

    # if cache_file.exists():
    #
    #     single_pulse_min_time = float(
    #         np.load(cache_file)
    #     )
    #
    #     print(
    #         "Loaded single XY_echo minimum time:",
    #         single_pulse_min_time,
    #     )
    #
    #     return single_pulse_min_time


    # --------------------------------------------------------
    # Optimize a single XY_echo pulse
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SINGLE XY_ECHO OPTIMIZATION")
    print("=" * 60)

    print(
        "Finding first minimum for a single XY_echo pulse..."
    )

    print_memory(
        "before single XY_echo optimization"
    )

    result = arr.optimize(
        ["XY_echo"],
        250,
        method="trust-constr",
        hessian=True,
        gradient=None,
        bounds=[
            (0, 1e7)
        ],
        results_dir=store.directory,
    )

    print_memory(
        "after single XY_echo optimization"
    )

    single_pulse_min_time = float(
        result.x[0]
    )

    print()
    print(
        "Single XY_echo objective:",
        float(result.fun),
    )

    print(
        "Single XY_echo minimum time:",
        single_pulse_min_time,
    )

    print(
        "Optimization success:",
        result.success,
    )

    print(
        "Optimization message:",
        result.message,
    )

    del result

    run_garbage_collection()

    print_memory(
        "after gc following single XY_echo optimization"
    )

    run_malloc_trim()

    print_memory(
        "after malloc_trim following single XY_echo optimization"
    )


    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    np.save(
        cache_file,
        single_pulse_min_time,
    )

    print(
        "Saved single XY_echo minimum time to:",
        cache_file.resolve(),
    )

    return single_pulse_min_time


# ============================================================
# Warm start
# ============================================================

def load_warm_start(
    source_ratio_dir,
    n_layers,
    target_sequence,
    target_bounds,
    target_single_pulse_min_time,
    rescale_xy=True,
):
    """
    Load an optimum from another (Rb, Omega/Delta) realization.

    XY_echo parameters are interpreted as interaction times and
    are rescaled according to

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


    # --------------------------------------------------------
    # Read source data
    # --------------------------------------------------------

    with np.load(
        summary_file
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


    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

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
    # Rescale ONLY XY_echo times
    # --------------------------------------------------------

    if rescale_xy:

        scale = (
            target_single_pulse_min_time
            / source_single_pulse_min_time
        )

        print()
        print(
            f"Rescaling XY_echo times by: "
            f"{scale}"
        )

        for i, gate in enumerate(
            target_sequence
        ):

            if gate == "XY_echo":
                x0[i] *= scale

            elif gate == "Rx":

                # Explicitly leave rotation unchanged.
                pass

    else:

        print()
        print(
            "XY_echo-time rescaling disabled."
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

def decode_task_id(
    task_id,
):

    n_rb = len(
        RB_VALUES
    )

    n_ratios = len(
        OMEGA_DELTA_RATIOS
    )

    n_tasks = (
        n_rb
        * n_ratios
        * MAX_LAYERS
    )

    if not 0 <= task_id < n_tasks:

        raise ValueError(
            f"task_id must be between "
            f"0 and {n_tasks - 1}, "
            f"got {task_id}"
        )

    tasks_per_rb = (
        n_ratios
        * MAX_LAYERS
    )

    rb_index = (
        task_id
        // tasks_per_rb
    )

    remainder = (
        task_id
        % tasks_per_rb
    )

    ratio_index = (
        remainder
        // MAX_LAYERS
    )

    n_layers = (
        remainder
        % MAX_LAYERS
        + 1
    )

    Rb = (
        RB_VALUES[
            rb_index
        ]
    )

    numerator, denominator = (
        OMEGA_DELTA_RATIOS[
            ratio_index
        ]
    )

    omega_over_delta = (
        numerator
        / denominator
    )

    return (
        Rb,
        numerator,
        denominator,
        omega_over_delta,
        n_layers,
    )


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
    ) = decode_task_id(
        task_id
    )


    # ========================================================
    # Print task information
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
        f"Number XY gates : "
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

    print(
        f"Memory checks   : "
        f"{MEMORY_DIAGNOSTICS}"
    )

    print(
        f"malloc_trim     : "
        f"{RUN_MALLOC_TRIM}"
    )

    print("=" * 60)

    print_memory(
        "at start of main()"
    )


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

    print_memory(
        "immediately before ConstructSquareArray"
    )

    arr = ConstructSquareArray(
        Lx=LX,
        Ly=LY,
        Rb=Rb,
        Omega=Omega,
        n=6,
        boundary="open",
        backend="quspin",
        use_reflections=True,
    )

    print_memory(
        "immediately after ConstructSquareArray"
    )


    # ========================================================
    # Results directories
    # ========================================================

    results_base_dir = Path(
        "results_XY/opt_XY"
    )

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
    # Find/load single XY_echo minimum
    # ========================================================

    single_pulse_min_time = (
        get_single_pulse_min_time(
            arr=arr,
            store=single_pulse_store,
            method=method,
            use_hessian=use_hessian,
            options=options,
        )
    )


    # ========================================================
    # XY interaction upper bound
    # ========================================================

    ub_XY = (
        2
        * single_pulse_min_time
    )

    print(
        f"Upper XY_echo bound: "
        f"{ub_XY}"
    )


    # ========================================================
    # Build gate sequence
    # ========================================================

    sequence, bounds = (
        gate_sequence(
            n_layers=n_layers,
            ub_XY=ub_XY,
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
    # Initial conditions
    # ========================================================

    initial_conditions = []

    n_warm_starts = 0


    # --------------------------------------------------------
    # Warm starts
    # --------------------------------------------------------

    if USE_WARM_START:

        for source_Rb in WARM_START_RB_VALUES:

            for (
                source_numerator,
                source_denominator,
            ) in WARM_START_OMEGA_DELTA_RATIOS:

                # --------------------------------------------
                # Skip target point itself
                # --------------------------------------------

                if (
                    np.isclose(
                        source_Rb,
                        Rb,
                    )
                    and source_numerator
                    == numerator
                    and source_denominator
                    == denominator
                ):

                    print()
                    print(
                        "Skipping warm start from target "
                        "point itself:"
                    )

                    print(
                        f"Rb = {source_Rb}, "
                        f"Omega/Delta = "
                        f"{source_numerator}/"
                        f"{source_denominator}"
                    )

                    continue


                # --------------------------------------------
                # Source directory
                # --------------------------------------------

                source_ratio_dir = (
                    parameter_directory(
                        base_dir=results_base_dir,
                        Lx=LX,
                        Ly=LY,
                        Rb=source_Rb,
                        numerator=source_numerator,
                        denominator=source_denominator,
                    )
                )

                source_summary_file = (
                    source_ratio_dir
                    / f"layers_{n_layers}"
                    / "summary.npz"
                )


                # --------------------------------------------
                # Skip missing source
                # --------------------------------------------

                if not source_summary_file.exists():

                    print()
                    print(
                        "Skipping missing warm-start source:"
                    )

                    print(
                        f"Rb = {source_Rb}, "
                        f"Omega/Delta = "
                        f"{source_numerator}/"
                        f"{source_denominator}"
                    )

                    print(
                        f"Missing file: "
                        f"{source_summary_file.resolve()}"
                    )

                    continue


                # --------------------------------------------
                # Load and rescale warm start
                # --------------------------------------------

                warm_start = (
                    load_warm_start(
                        source_ratio_dir=source_ratio_dir,
                        n_layers=n_layers,
                        target_sequence=sequence,
                        target_bounds=bounds,
                        target_single_pulse_min_time=(
                            single_pulse_min_time
                        ),
                        rescale_xy=(
                            WARM_START_RESCALE_XY
                        ),
                    )
                )

                initial_conditions.append(
                    warm_start
                )

                n_warm_starts += 1


    # --------------------------------------------------------
    # Latin-hypercube initial conditions
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

        initial_conditions.extend(
            lhs_conditions
        )


    # --------------------------------------------------------
    # Convert to array
    # --------------------------------------------------------

    initial_conditions = (
        np.asarray(
            initial_conditions,
            dtype=float,
        )
    )

    if len(
        initial_conditions
    ) == 0:

        raise RuntimeError(
            "No initial conditions were generated."
        )


    # --------------------------------------------------------
    # Initial-condition summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("INITIAL CONDITIONS")
    print("=" * 60)

    print(
        "Total initial conditions:",
        len(initial_conditions),
    )

    if USE_WARM_START:

        print(
            "Warm-start conditions:",
            n_warm_starts,
        )

        print(
            "Warm-start Rb values:",
            WARM_START_RB_VALUES,
        )

        print(
            "Warm-start Omega/Delta ratios:",
            WARM_START_OMEGA_DELTA_RATIOS,
        )

        print(
            "Rescale XY_echo times:",
            WARM_START_RESCALE_XY,
        )

        print(
            "Rx rotations rescaled:",
            False,
        )

    print_memory(
        "after generation of initial conditions"
    )


    # ========================================================
    # Run optimizations
    # ========================================================

    # Do not retain all scipy OptimizeResult objects.
    # Keep only the best objective and its parameter vector.

    best_fun = np.inf
    best_x = None

    n_optimizations = len(
        initial_conditions
    )

    for i, x0 in enumerate(
        initial_conditions
    ):

        print()
        print("=" * 60)

        print(
            f"Optimization "
            f"{i + 1}/"
            f"{n_optimizations}"
        )

        if (
            USE_WARM_START
            and i < n_warm_starts
        ):

            print(
                "Initial condition: "
                "WARM START"
            )

        else:

            print(
                "Initial condition: "
                "LHS"
            )

        print("=" * 60)


        # ----------------------------------------------------
        # Memory before
        # ----------------------------------------------------

        print_memory(
            f"before optimization {i + 1}"
        )


        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Memory immediately after
        # ----------------------------------------------------

        print_memory(
            f"immediately after optimization {i + 1}"
        )


        # ----------------------------------------------------
        # Extract required data only
        # ----------------------------------------------------

        current_fun = float(
            result.fun
        )

        current_x = np.array(
            result.x,
            dtype=float,
            copy=True,
        )

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


        # ----------------------------------------------------
        # Update best result
        # ----------------------------------------------------

        if (
            best_x is None
            or current_fun < best_fun
        ):

            best_fun = current_fun

            best_x = current_x.copy()

            print(
                "New best result:",
                best_fun,
            )


        # ----------------------------------------------------
        # Explicitly release OptimizeResult
        # ----------------------------------------------------

        del result
        del current_x


        # ----------------------------------------------------
        # Garbage collection
        # ----------------------------------------------------

        run_garbage_collection()

        print_memory(
            f"after gc.collect(), "
            f"optimization {i + 1}"
        )


        # ----------------------------------------------------
        # glibc malloc_trim
        # ----------------------------------------------------

        run_malloc_trim()

        print_memory(
            f"after malloc_trim(), "
            f"optimization {i + 1}"
        )

        print(
            "-" * 60
        )


    # ========================================================
    # Save compact summary
    # ========================================================

    if best_x is None:

        raise RuntimeError(
            "No optimization result was obtained."
        )

    summary_file = (
        layer_dir
        / "summary.npz"
    )

    np.savez(
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
        best_fun=best_fun,
        best_x=best_x,
        single_pulse_min_time=(
            single_pulse_min_time
        ),
        ub_XY=ub_XY,
    )


    # ========================================================
    # Summary
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
        f"XY gates       : "
        f"{n_layers}"
    )

    print(
        f"Gate sequence  : "
        f"{sequence}"
    )

    print(
        f"Single XY tmin : "
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
        f"Summary saved  : "
        f"{summary_file.resolve()}"
    )

    print_memory(
        "at end of main()"
    )


if __name__ == "__main__":
    main()