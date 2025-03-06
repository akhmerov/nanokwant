import numpy as np
from nanokwant.nanokwant import (
    _hamiltonian_dtype,
    matrix_hamiltonian,
    hamiltonian,
    _to_banded,
)
from scipy.linalg import eig_banded


def test__hamiltonian_dtype():
    test_cases = [
        (
            {
                0: {"mu": np.eye(2, dtype=np.float64)},
            },
            {"mu": 1.0},
            np.float64,
        ),
        (
            {
                0: {"mu": np.eye(2, dtype=np.float64)},
            },
            {"mu": 1.0 + 1j},
            np.complex128,
        ),
        (
            {
                0: {"mu": np.eye(2, dtype=np.float64)},
            },
            {"mu": lambda x: 1.0},
            np.float64,
        ),
        (
            {
                0: {"mu": np.eye(2, dtype=np.float64)},
            },
            {"mu": lambda x: 1.0 + 1j},
            np.complex128,
        ),
        (
            {
                0: {
                    "mu": np.eye(2, dtype=np.float64),
                    "Ez": np.diag([1, -1]).astype(np.complex128),
                },
            },
            {"mu": 1.0, "Ez": 1.0 + 1j},
            np.complex128,
        ),
    ]

    for system, params, expected_dtype in test_cases:
        dtype = _hamiltonian_dtype(system, params)
        assert dtype == expected_dtype, f"Expected {expected_dtype}, but got {dtype}"


def test_matrix_hamiltonian():
    system = {
        0: {
            "mu": np.eye(2),
            "Ez": np.diag([1, -1]),
        },
        1: {
            "t": np.eye(2),
        },
    }
    num_sites = 3
    params = {
        "mu": 1.0,
        "Ez": (lambda x: 1.0 + 1j * (x == 0)),
        "t": (lambda x: x + 1),
    }
    expected_H = np.array(
        [
            [2 + 1j, 0, 1, 0, 0, 0],
            [0, -1j, 0, 1, 0, 0],
            [1, 0, 2, 0, 2, 0],
            [0, 1, 0, 0, 0, 2],
            [0, 0, 2, 0, 2, 0],
            [0, 0, 0, 2, 0, 0],
        ]
    )
    H = matrix_hamiltonian(system, num_sites, params)
    np.testing.assert_equal(H, expected_H)


def test_to_banded():
    for a in (
        np.random.rand(5, 3),
        np.random.rand(3, 5),
        np.random.rand(4, 4),
    ):
        ab = _to_banded(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                assert ab[a.shape[1] - 1 - j + i, j] == a[i, j], (i, j)


def test_hamiltonian():
    system = {
        0: {
            "mu": np.eye(2),
            "Ez": np.array([[0, 1], [1, 0]]),
        },
        1: {
            "t": np.eye(2),
        },
    }
    num_sites = 3
    params = {
        "mu": 2.0,
        "Ez": lambda x: np.sin(x),
        "t": 1.0,
    }
    H, (l, u) = hamiltonian(system, num_sites, params)  # noqa: E741
    eigvals_banded = eig_banded(H[: u + 1], eigvals_only=True)
    H_matrix = matrix_hamiltonian(system, num_sites, params)
    eigvals_matrix = np.linalg.eigvalsh(H_matrix)
    np.testing.assert_allclose(eigvals_banded, eigvals_matrix)
