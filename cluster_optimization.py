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

method = "L-BFGS-B"
use_hessian = False

# method = "trust-constr"
# use_hessian = True

options = {
    "maxiter": 1000,
}

save_best = True


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


# ============================================================
# Gate sequence
# ============================================================

def gate_sequence(n_layers, ub_Ising):
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
            bounds.append((-np.pi, np.pi))

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

def sample_lhs(bounds, n_samples, seed=None):
    """
    Generate Latin-hypercube initial conditions within bounds.
    """

    bounds = np.asarray(bounds, dtype=float)

    lower = bounds[:, 0]
    upper = bounds[:, 1]

    sampler = qmc.LatinHypercube(
        d=len(bounds),
        seed=seed,
    )

    samples = sampler.random(n=n_samples)

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

    store_dir = Path(store.directory)
    store_dir.mkdir(parents=True, exist_ok=True)

    cache_file = store_dir / "single_pulse_min_time.npy"

    if cache_file.exists():

        single_pulse_min_time = float(
            np.load(cache_file)
        )

        print(
            "Loaded single-pulse minimum time:",
            single_pulse_min_time,
        )

        return single_pulse_min_time

    print("Optimizing single Iz_echo pulse...")

    result = arr.optimize(
        ["Iz_echo"],
        1,
        method=method,
        hessian=use_hessian,
        bounds=[(0, 1e4)],
        options=options,
        results_dir=store.directory,
    )

    single_pulse_min_time = float(result.x[0])

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
# Decode Slurm task
# ============================================================

def decode_task_id(task_id):
    """
    Task ordering:

    For each Rb:
        for each Omega/Delta:
            layers 1,...,MAX_LAYERS

    With:
        3 Rb values
        3 Omega/Delta values
        5 layers

    total = 45 tasks
    """

    n_rb = len(RB_VALUES)
    n_ratios = len(OMEGA_DELTA_RATIOS)

    n_tasks = (
        n_rb
        * n_ratios
        * MAX_LAYERS
    )

    if not 0 <= task_id < n_tasks:
        raise ValueError(
            f"task_id must be between 0 and {n_tasks - 1}, "
            f"got {task_id}"
        )

    tasks_per_rb = (
        n_ratios
        * MAX_LAYERS
    )

    rb_index = task_id // tasks_per_rb

    remainder = task_id % tasks_per_rb

    ratio_index = remainder // MAX_LAYERS

    n_layers = remainder % MAX_LAYERS + 1

    Rb = RB_VALUES[rb_index]

    numerator, denominator = (
        OMEGA_DELTA_RATIOS[ratio_index]
    )

    omega_over_delta = numerator / denominator

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
    ) = decode_task_id(task_id)

    print("=" * 60)
    print(f"Task ID         : {task_id}")
    print(f"Geometry        : {LX}x{LY}")
    print(f"Rb              : {Rb}")
    print(
        f"Omega / Delta   : "
        f"{numerator}/{denominator} "
        f"= {omega_over_delta}"
    )
    print(f"Number layers   : {n_layers}")
    print("=" * 60)


    # --------------------------------------------------------
    # Physical parameters
    # --------------------------------------------------------

    Delta = 1.0
    Omega = omega_over_delta * Delta

    print(f"Delta           : {Delta}")
    print(f"Omega           : {Omega}")


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

    geometry_dir = (
        Path("results_Ising")
        / f"{LX}x{LY}"
    )

    rb_dir = (
        geometry_dir
        / f"Rb_{format_float_for_path(Rb)}"
    )

    ratio_dir = (
        rb_dir
        / f"OmegaDelta_{numerator}_{denominator}"
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
    print("Results directory:")
    print(layer_dir.resolve())


    # --------------------------------------------------------
    # Result stores
    # --------------------------------------------------------

    # Single-pulse minimum depends on geometry, Rb and Omega/Delta,
    # but not on the number of layers.
    single_pulse_store = OptimizationResultStore(
        directory=ratio_dir
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

    ub_Ising = 1.5 * single_pulse_min_time

    print(
        f"Upper Ising bound: {ub_Ising}"
    )


    # --------------------------------------------------------
    # Build sequence
    # --------------------------------------------------------

    sequence, bounds = gate_sequence(
        n_layers=n_layers,
        ub_Ising=ub_Ising,
    )

    print()
    print("Sequence:")
    print(sequence)

    print()
    print("Bounds:")
    print(bounds)


    # --------------------------------------------------------
    # Latin-hypercube initial conditions
    # --------------------------------------------------------

    if LHS_SEED_BASE is None:
        seed = None
    else:
        seed = LHS_SEED_BASE + task_id

    initial_conditions = sample_lhs(
        bounds=bounds,
        n_samples=N_INITIAL_CONDITIONS,
        seed=seed,
    )

    print()
    print(
        "Number of initial conditions:",
        len(initial_conditions),
    )


    # --------------------------------------------------------
    # Run optimizations
    # --------------------------------------------------------

    best_result = None

    for i, x0 in enumerate(initial_conditions):

        print()
        print("-" * 60)
        print(
            f"Optimization "
            f"{i + 1}/{N_INITIAL_CONDITIONS}"
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
            or result.fun < best_result.fun
        ):
            best_result = result

            print(
                "New best result:",
                best_result.fun,
            )


    # --------------------------------------------------------
    # Save compact summary
    # --------------------------------------------------------

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
        omega_over_delta=omega_over_delta,
        n_layers=n_layers,
        best_fun=best_result.fun,
        best_x=best_result.x,
        single_pulse_min_time=single_pulse_min_time,
        ub_Ising=ub_Ising,
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

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