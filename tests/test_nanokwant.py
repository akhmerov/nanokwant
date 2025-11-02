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
        ab, (l, u) = _to_banded(a, shrink=False)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                assert ab[a.shape[1] - 1 - j + i, j] == a[i, j], (i, j)


def test_to_banded_shrink_equivalence():
    """Ensure `_to_banded(..., shrink=True)` detects bandwidth correctly.

    Mirrors the previous behavior used in scattering tests.
    """
    A = np.array(
        [
            [1, 2, 0, 0],
            [3, 4, 5, 0],
            [0, 6, 7, 8],
            [0, 0, 9, 10],
        ],
        dtype=float,
    )

    A_band, (l, u) = _to_banded(A)

    assert l == 1
    assert u == 1
    assert A_band.shape == (3, 4)

    # Verify the banded format
    for i in range(4):
        for j in range(4):
            if abs(i - j) <= 1:
                assert abs(A_band[u + i - j, j] - A[i, j]) < 1e-10


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


def test_array_parameters_infer_num_sites():
    system = {
        0: {"mu": np.eye(1)},
        1: {"t": np.eye(1)},
    }
    # For N sites, onsite array length N, hopping length N-1
    mu = np.array([1.0, 2.0, 3.0])  # length 3 -> num_sites should be 3
    t = np.array([0.5, 0.6])  # length 2 -> N-1 consistent
    params = {"mu": mu, "t": t}
    H_banded, (l, u) = hamiltonian(system, None, params)  # noqa: E741
    H_matrix = matrix_hamiltonian(system, None, params)
    eigvals_banded = eig_banded(H_banded[: u + 1], eigvals_only=True)
    eigvals_matrix = np.linalg.eigvalsh(H_matrix)
    np.testing.assert_allclose(np.sort(eigvals_banded), np.sort(eigvals_matrix))


def test_array_parameter_inconsistency():
    system = {0: {"mu": np.eye(1)}, 1: {"t": np.eye(1)}}
    mu = np.array([1.0, 2.0, 3.0])
    t = np.array([0.5, 0.6, 0.7])  # should be length 2 for 3 sites
    params = {"mu": mu, "t": t}
    try:
        hamiltonian(system, None, params)
    except ValueError as e:
        assert "length" in str(e)
    else:
        raise AssertionError("Expected ValueError for inconsistent parameter lengths")


def test_hamiltonian_eig_banded_format():
    """Test that eig_banded format produces correct results."""
    system = {
        0: {
            "mu": np.eye(2),
            "Ez": np.array([[0, 1], [1, 0]]),
        },
        1: {
            "t": np.eye(2),
        },
    }
    num_sites = 5
    params = {
        "mu": 2.0,
        "Ez": lambda x: np.sin(x),
        "t": 1.0,
    }

    # General format (for solve_banded)
    H_general, (l, u) = hamiltonian(system, num_sites, params, format="general")  # noqa: E741

    # eig_banded format (optimized, only upper bands)
    H_eig, (l_eig, u_eig) = hamiltonian(system, num_sites, params, format="eig_banded")  # noqa: E741
    assert l_eig == 0, "eig_banded format should have l=0"

    # The eig_banded result must equal the top (u+1) rows of the general band
    # (i.e. it should be a slice of the full banded matrix).
    np.testing.assert_equal(H_eig, H_general[: u + 1])

    # Eigenvalues: compare optimized band eigenvalues to dense matrix eigenvalues
    eigvals_eig = eig_banded(H_eig, eigvals_only=True, lower=False)
    H_matrix = matrix_hamiltonian(system, num_sites, params)
    eigvals_matrix = np.linalg.eigvalsh(H_matrix)
    np.testing.assert_allclose(np.sort(eigvals_eig), np.sort(eigvals_matrix))


def test_trimming_eig_banded():
    """Test that the lower diagonal part of the onsite terms is trimmed."""
    system = {
        0: {"a": np.ones((2, 2))},
    }
    num_sites = 3
    params = {"a": 1.0}
    H_band, (l, u) = hamiltonian(
        system, num_sites, params, hermitian=True, format="eig_banded"
    )
    assert (l, u) == (0, 1)
    assert H_band.shape[0] == 2


def test_shrink_no_diagonal():
    """If there are no hop=0 (onsite) terms, the returned band should be
    correctly shrunk: bandwidths non-negative and no all-zero outer rows.

    This guards against regressions where outer rows remain zero or l/u
    become negative after trimming.
    """
    # System with only nearest-neighbor hopping (no onsite terms)
    system = {
        1: {"t": np.array([[0.0, 1.0], [1.0, 0.0]])},
    }
    num_sites = 4
    params = {"t": 1.0}

    H_band, (l, u) = hamiltonian(
        system, num_sites, params, hermitian=False, format="general"
    )

    # Bandwidths must be non-negative integers
    assert isinstance(l, (int, np.integer)) and isinstance(u, (int, np.integer))
    assert l >= 0 and u >= 0

    # The returned band should not have all-zero top rows (longest range hopping)
    assert np.any(H_band[0])
