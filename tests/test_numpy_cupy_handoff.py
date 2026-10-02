"""Tests for the standalone HeteroGuard NumPy/CuPy example.

These tests require NumPy and pytest only. A GPU is deliberately not required:
the assurance mechanics are tested independently of CuPy availability.
"""

import numpy as np

from examples.numpy_cupy_handoff import (
    finite_guard,
    guarded_matmul,
    probe_matmul,
    projected_matmul_witness,
)


def matrices(size: int = 24, seed: int = 123):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    return a, b


def test_probe_accepts_matching_backend():
    result = probe_matmul(np.matmul)
    assert result.passed
    assert result.error <= result.tolerance


def test_projected_witness_accepts_correct_product():
    a, b = matrices()
    c = a @ b

    error = projected_matmul_witness(a, b, c)

    assert error < 1e-5


def test_projected_witness_detects_corrupted_product():
    a, b = matrices()
    c = a @ b
    c = c.copy()
    c[0, 0] += 1_000.0

    error = projected_matmul_witness(a, b, c)

    assert error > 1e-3


def test_guarded_matmul_accepts_valid_candidate():
    a, b = matrices()

    result = guarded_matmul(
        a,
        b,
        candidate=np.matmul,
        candidate_name="test-candidate",
    )

    assert not result.fallback_used
    assert result.backend == "test-candidate"
    np.testing.assert_allclose(result.value, a @ b)


def test_guarded_matmul_falls_back_when_witness_fails():
    a, b = matrices()

    def bad_candidate(x, y):
        result = x @ y
        result = result.copy()
        result[0, 0] += 1_000.0
        return result

    result = guarded_matmul(
        a,
        b,
        candidate=bad_candidate,
        candidate_name="bad-candidate",
    )

    assert result.fallback_used
    assert result.backend == "cpu-fallback"
    assert result.witness_error is not None
    np.testing.assert_allclose(result.value, a @ b)


def test_guarded_matmul_falls_back_on_backend_exception():
    a, b = matrices()

    def unavailable_candidate(x, y):
        raise RuntimeError("backend unavailable")

    result = guarded_matmul(
        a,
        b,
        candidate=unavailable_candidate,
        candidate_name="unavailable",
    )

    assert result.fallback_used
    assert result.backend == "cpu-fallback"
    assert result.witness_error is None
    assert "backend unavailable" in result.reason
    np.testing.assert_allclose(result.value, a @ b)


def test_finite_guard_rejects_nan_and_inf():
    assert finite_guard(np.array([1.0, 2.0]))
    assert not finite_guard(np.array([1.0, np.nan]))
    assert not finite_guard(np.array([1.0, np.inf]))
