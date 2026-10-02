"""Minimal HeteroGuard demonstration using NumPy and optional CuPy.

The example demonstrates four ideas:

1. Probe a candidate backend before trusting it.
2. Dispatch expensive work to that backend only after the probe passes.
3. Verify the result with a cheaper independent witness.
4. Fall back to NumPy if the candidate backend is unavailable or rejected.

For C = A @ B, the witness checks:

    C @ r ~= A @ (B @ r)

for a deterministic random vector r.

For square n x n matrices, the witness uses matrix-vector work (O(n^2))
rather than repeating the full matrix-matrix multiply (O(n^3)).

Run:

    python examples/numpy_cupy_handoff.py
    python examples/numpy_cupy_handoff.py --inject-error

CuPy is optional. Without it, the example demonstrates CPU fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
Matmul = Callable[[Array, Array], Array]


@dataclass(frozen=True)
class ProbeResult:
    passed: bool
    error: float
    tolerance: float


@dataclass(frozen=True)
class HandoffResult:
    value: Array
    backend: str
    witness_error: float | None
    fallback_used: bool
    reason: str


def relative_l2_error(actual: Array, expected: Array) -> float:
    """Return ||actual - expected||_2 / max(||expected||_2, eps)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    denominator = max(float(np.linalg.norm(expected)), np.finfo(float).eps)
    return float(np.linalg.norm(actual - expected) / denominator)


def finite_guard(value: Array) -> bool:
    """Minimal invariant guard: accepted results must contain finite values."""
    return bool(np.isfinite(np.asarray(value)).all())


def projected_matmul_witness(
    a: Array,
    b: Array,
    c: Array,
    *,
    seed: int = 7,
) -> float:
    """Witness C = A @ B using C @ r ~= A @ (B @ r)."""
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise ValueError("witness expects rank-2 arrays")
    if a.shape[1] != b.shape[0]:
        raise ValueError("A and B have incompatible shapes")
    if c.shape != (a.shape[0], b.shape[1]):
        raise ValueError("candidate result has the wrong shape")

    rng = np.random.default_rng(seed)
    dtype = np.result_type(a.dtype, b.dtype, c.dtype)
    r = rng.standard_normal(b.shape[1]).astype(dtype, copy=False)

    lhs = c @ r
    rhs = a @ (b @ r)
    return relative_l2_error(lhs, rhs)


def probe_matmul(
    candidate: Matmul,
    *,
    reference: Matmul = np.matmul,
    size: int = 48,
    tolerance: float = 5e-5,
    seed: int = 11,
) -> ProbeResult:
    """Compare a candidate matmul backend with NumPy on a small test case."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)

    expected = np.asarray(reference(a, b))
    actual = np.asarray(candidate(a, b))
    error = relative_l2_error(actual, expected)

    return ProbeResult(
        passed=finite_guard(actual) and error <= tolerance,
        error=error,
        tolerance=tolerance,
    )


def guarded_matmul(
    a: Array,
    b: Array,
    *,
    candidate: Matmul,
    candidate_name: str,
    fallback: Matmul = np.matmul,
    witness_tolerance: float = 5e-5,
    inject_error: bool = False,
) -> HandoffResult:
    """Run a candidate backend, witness its result, and fall back on failure."""
    try:
        candidate_value = np.asarray(candidate(a, b))

        # Demonstration-only fault injection. It lets the example prove that
        # the witness is active rather than merely logging successful runs.
        if inject_error:
            candidate_value = candidate_value.copy()
            candidate_value[0, 0] += np.asarray(
                1_000.0, dtype=candidate_value.dtype
            )

        if not finite_guard(candidate_value):
            raise ValueError("candidate produced non-finite values")

        witness_error = projected_matmul_witness(a, b, candidate_value)

        if witness_error <= witness_tolerance:
            return HandoffResult(
                value=candidate_value,
                backend=candidate_name,
                witness_error=witness_error,
                fallback_used=False,
                reason="candidate accepted",
            )

        fallback_value = np.asarray(fallback(a, b))
        return HandoffResult(
            value=fallback_value,
            backend="cpu-fallback",
            witness_error=witness_error,
            fallback_used=True,
            reason=(
                f"witness error {witness_error:.3e} exceeded "
                f"tolerance {witness_tolerance:.3e}"
            ),
        )

    except Exception as exc:
        fallback_value = np.asarray(fallback(a, b))
        return HandoffResult(
            value=fallback_value,
            backend="cpu-fallback",
            witness_error=None,
            fallback_used=True,
            reason=f"{type(exc).__name__}: {exc}",
        )


def cupy_backend() -> tuple[Matmul | None, str]:
    """Return a CuPy matmul adapter if a usable CUDA runtime is available."""
    try:
        import cupy as cp

        # Force an actual device operation. Successful import alone does not
        # prove that the CUDA runtime is usable.
        cp.asarray([1.0], dtype=cp.float32).sum().item()

        def matmul(a: Array, b: Array) -> Array:
            ga = cp.asarray(a)
            gb = cp.asarray(b)
            return cp.asnumpy(ga @ gb)

        return matmul, "cupy-gpu"

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def print_status(label: str, message: str) -> None:
    print(f"[{label:<8}] {message}")


def run_demo(size: int, inject_error: bool) -> int:
    print("HeteroGuard NumPy/CuPy handoff demo")
    print("-----------------------------------")

    candidate, backend_status = cupy_backend()

    if candidate is None:
        print_status("PROBE", f"GPU backend unavailable ({backend_status})")
        print_status("FALLBACK", "using NumPy CPU path")

        rng = np.random.default_rng(21)
        a = rng.standard_normal((size, size), dtype=np.float32)
        b = rng.standard_normal((size, size), dtype=np.float32)
        value = a @ b

        guard_ok = finite_guard(value)
        print_status("GUARD", f"finite result {'PASS' if guard_ok else 'FAIL'}")
        print_status("DECISION", "ACCEPT CPU RESULT")
        return 0 if guard_ok else 1

    probe = probe_matmul(candidate)
    print_status(
        "PROBE",
        f"CuPy GPU {'PASS' if probe.passed else 'FAIL'} "
        f"(relative error={probe.error:.3e}, tolerance={probe.tolerance:.1e})",
    )

    if not probe.passed:
        print_status("FALLBACK", "probe rejected GPU backend; using NumPy")
        return 0

    rng = np.random.default_rng(21)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)

    print_status("DISPATCH", f"{size}x{size} matrix multiply -> GPU")

    result = guarded_matmul(
        a,
        b,
        candidate=candidate,
        candidate_name="cupy-gpu",
        inject_error=inject_error,
    )

    if inject_error:
        print_status(
            "INJECT",
            "perturbed candidate result for failure demonstration",
        )

    if result.witness_error is not None:
        witness_passed = not result.fallback_used
        print_status(
            "WITNESS",
            f"{'PASS' if witness_passed else 'FAIL'} "
            f"(relative error={result.witness_error:.3e})",
        )

    if result.fallback_used:
        print_status("FALLBACK", result.reason)

    guard_ok = finite_guard(result.value)
    print_status("GUARD", f"finite result {'PASS' if guard_ok else 'FAIL'}")
    print_status(
        "DECISION",
        "ACCEPT CPU RESULT" if result.fallback_used else "ACCEPT GPU RESULT",
    )

    return 0 if guard_ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="square matrix dimension for the demonstration (default: 1024)",
    )
    parser.add_argument(
        "--inject-error",
        action="store_true",
        help="corrupt the candidate result so the witness triggers CPU fallback",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_demo(args.size, args.inject_error))
