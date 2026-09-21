# cluster_optimization.py

import os
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
    (1, 10),
]

MAX_LAYERS = 5

N_INITIAL_CONDITIONS = 1000
LHS_SEED_BASE = None

# method = "L-BFGS-B"
# use_hessian = False

method = "SLSQP"
use_hessian = False

# method = "trust-constr"
# use_hessian = True

options = {
    "maxiter": 1000,
}

save_best = True


# ============================================================
# Warm-start settings
# ============================================================

# If False, the script behaves as before and only uses
# Latin-hypercube initial conditions.
USE_WARM_START = True


# Source point from which the current optimum should be loaded.
#
# Example:
#
#     WARM_START_RB = 2.0
#     WARM_START_OMEGA_DELTA = (1, 3)
#
# loads from
#
# results_Ising/
#     4x4/
#         Rb_2/
#             OmegaDelta_1_3/
#                 layers_N/
#                     summary.npz
#

WARM_START_RB_VALUES = [
    1.5,
    2.0,
    2.5,
]

WARM_START_OMEGA_DELTA_RATIOS = [
    (1, 2),
    (1, 3),
    (1, 10),
]


# If True:
#
#     warm-start optimum
#     + N_INITIAL_CONDITIONS LHS points
#
# are optimized.
#
# If False:
#
#     only the warm-start optimum
#
# is optimized.
WARM_START_INCLUDE_LHS = True


# If True, rescale only Iz_echo and Ix_echo durations:
#
#       t_target
#       -------- = target single-pulse minimum time
#       t_source   source single-pulse minimum time
#
# Rx angles remain unchanged.
#
# If False, the old parameters are copied directly.
WARM_START_RESCALE_ISING = True


# ============================================================
# Helpers
# ============================================================

def format_float_for_path(x):
    """
    Convert e.g.

        1.5 -> '1p5'
        2.0 -> '2'
        2.5 -> '2p5'
    """

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
    """
    Return the result directory corresponding to

        geometry
        Rb
        Omega / Delta
    """

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
    """
    Project a parameter vector onto the optimization bounds.
    """

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
    ub_Ising,
):
    """
    Generate

        Iz_echo, Ix_echo,
        Rx, Iz_echo, Ix_echo,
        Rx, Iz_echo, Ix_echo,
        ...

    Each layer contains Iz_echo and Ix_echo.
    Every layer except the first is preceded by Rx.
    """

    sequence = []
    bounds = []

    for layer in range(n_layers):

        if layer > 0:
            sequence.append("Rx")
            bounds.append(
                (-np.pi, np.pi)
            )

        sequence.extend([
            "Iz_echo",
            "Ix_echo",
        ])

        bounds.extend([
            (0, ub_Ising),
            (0, ub_Ising),
        ])

    return sequence, bounds


# ============================================================
# Latin hypercube sampling
# ============================================================

def sample_lhs(
    bounds,
    n_samples,
    seed=None,
):
    """
    Generate Latin-hypercube initial conditions within bounds.
    """

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

def get_single_pulse_min_time(
    arr,
    store,
    method,
    use_hessian,
    options,
):
    """
    Calculate the optimal duration of a single Iz_echo pulse
    once and cache it in the corresponding parameter directory.
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

    if cache_file.exists():

        single_pulse_min_time = float(
            np.load(cache_file)
        )

        print(
            "Loaded single-pulse minimum time:",
            single_pulse_min_time,
        )

        return single_pulse_min_time

    print(
        "Optimizing single Iz_echo pulse..."
    )

    result = arr.optimize(
        ["Iz_echo"],
        1,
        method=method,
        hessian=use_hessian,
        bounds=[
            (0, 1e4)
        ],
        options=options,
        results_dir=store.directory,
    )

    single_pulse_min_time = float(
        result.x[0]
    )

    np.save(
        cache_file,
        single_pulse_min_time,
    )

    print(
        "Single-pulse minimum time:",
        single_pulse_min_time,
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
    rescale_ising=True,
):
    """
    Load the current optimum from another Rb / Omega-Delta point.

    Expected source:

        source_ratio_dir/
            layers_N/
                summary.npz

    The number of layers must match the current optimization.

    If rescale_ising=True, only Iz_echo and Ix_echo parameters
    are rescaled according to

        t_new = t_old
                * target_single_pulse_min_time
                / source_single_pulse_min_time

    Rx parameters are unchanged.
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
    # Check dimensionality
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
    # Optional Ising-time rescaling
    # --------------------------------------------------------

    if rescale_ising:

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
                "Iz_echo",
                "Ix_echo",
            ):

                x0[i] *= scale

    else:

        print()
        print(
            "Ising-time rescaling disabled."
        )


    # --------------------------------------------------------
    # Enforce target bounds
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
    # Print warm start
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
    """
    Task ordering:

    For each Rb:
        for each Omega/Delta:
            layers 1,...,MAX_LAYERS

    Total:

        len(RB_VALUES)
        * len(OMEGA_DELTA_RATIOS)
        * MAX_LAYERS
    """

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

    # --------------------------------------------------------
    # Slurm task ID
    # --------------------------------------------------------

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

    print("=" * 60)


    # --------------------------------------------------------
    # Physical parameters
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Construct array
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Results directories
    # --------------------------------------------------------

    results_base_dir = Path(
        "results_Ising"
    )

    geometry_dir = (
        results_base_dir
        / f"{LX}x{LY}"
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


    # --------------------------------------------------------
    # Result stores
    # --------------------------------------------------------

    # Single-pulse minimum depends on geometry, Rb and
    # Omega/Delta, but not on the number of layers.
    single_pulse_store = (
        OptimizationResultStore(
            directory=ratio_dir
        )
    )

    # Layer-specific optimizations go here.
    store = OptimizationResultStore(
        directory=layer_dir
    )


    # --------------------------------------------------------
    # Find/load single-pulse minimum
    # --------------------------------------------------------

    single_pulse_min_time = (
        get_single_pulse_min_time(
            arr=arr,
            store=single_pulse_store,
            method=method,
            use_hessian=use_hessian,
            options=options,
        )
    )


    # --------------------------------------------------------
    # Ising interaction upper bound
    # --------------------------------------------------------

    ub_Ising = (
        1.5
        * single_pulse_min_time
    )

    print(
        f"Upper Ising bound: "
        f"{ub_Ising}"
    )


    # --------------------------------------------------------
    # Build sequence
    # --------------------------------------------------------

    sequence, bounds = (
        gate_sequence(
            n_layers=n_layers,
            ub_Ising=ub_Ising,
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

    # --------------------------------------------------------
    # Warm-start initial conditions
    # --------------------------------------------------------

    n_warm_starts = 0

    if USE_WARM_START:

        for source_Rb in WARM_START_RB_VALUES:

            for (
                    source_numerator,
                    source_denominator,
            ) in WARM_START_OMEGA_DELTA_RATIOS:

                # Skip the target point itself
                if (
                        np.isclose(source_Rb, Rb)
                        and source_numerator == numerator
                        and source_denominator == denominator
                ):
                    print()
                    print(
                        "Skipping warm start from target point itself:"
                    )
                    print(
                        f"Rb = {source_Rb}, "
                        f"Omega/Delta = "
                        f"{source_numerator}/{source_denominator}"
                    )
                    continue

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

                # If no optimization exists for this source point,
                # simply skip it.
                if not source_summary_file.exists():
                    print()
                    print(
                        "Skipping missing warm-start source:"
                    )

                    print(
                        f"Rb = {source_Rb}, "
                        f"Omega/Delta = "
                        f"{source_numerator}/{source_denominator}"
                    )

                    print(
                        f"Missing file: "
                        f"{source_summary_file.resolve()}"
                    )

                    continue

                warm_start = (
                    load_warm_start(
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
            "Rescale Ising times:",
            WARM_START_RESCALE_ISING,
        )

    # ========================================================
    # Run optimizations
    # ========================================================

    best_result = None

    n_optimizations = len(
        initial_conditions
    )

    for i, x0 in enumerate(
        initial_conditions
    ):

        print()
        print("-" * 60)

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

        print("-" * 60)

        result = arr.optimize(
            sequence,
            x0,
            method=method,
            hessian=use_hessian,
            bounds=bounds,
            options=options,
            results_dir=(
                store.directory
                if save_best
                else None
            ),
        )

        if (
            best_result is None
            or result.fun
            < best_result.fun
        ):

            best_result = result

            print(
                "New best result:",
                best_result.fun,
            )


    # ========================================================
    # Save compact summary
    # ========================================================

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
        best_fun=best_result.fun,
        best_x=best_result.x,
        single_pulse_min_time=(
            single_pulse_min_time
        ),
        ub_Ising=ub_Ising,
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
        f"Layers         : "
        f"{n_layers}"
    )

    print(
        f"Best objective : "
        f"{best_result.fun}"
    )

    print(
        "Best parameters:"
    )

    print(
        best_result.x
    )

    print(
        f"Summary saved  : "
        f"{summary_file.resolve()}"
    )


if __name__ == "__main__":
    main()