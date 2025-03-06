import numpy as np
from typing import Union, Callable

HamiltonianType = dict[int, dict[str, np.ndarray]]


def _hamiltonian_dtype(
    system: HamiltonianType, params: dict[str, complex | Callable]
) -> np.dtype:
    """Determine the common dtype of all term matrices and all parameter values."""
    term_dtypes = [
        term_matrix.dtype for terms in system.values() for term_matrix in terms.values()
    ]
    param_dtypes = [
        np.asarray(value).dtype if not callable(value) else np.asarray(value(0)).dtype
        for value in params.values()
    ]
    return np.result_type(*term_dtypes, *param_dtypes)


def _to_banded(a: np.ndarray) -> np.ndarray:
    """Convert a full 2D array to banded format.

    The diagonal ordered format is defined as

        ab[u + i - j, j] == a[i, j]
    """
    (m, n) = a.shape
    m += n - 1
    ab = np.zeros((m, n), dtype=a.dtype)
    for i in range(n):
        ab[n - 1 - i : m - i, i] = a[:, i]
    return ab


def hamiltonian(
    system: HamiltonianType,
    num_sites: int,
    params: Union[complex, Callable],
    hermitian: bool = True,
) -> np.ndarray:
    """Generate the finite system Hamiltonian in band matrix format.

    The diagonal ordered format defined as

        ab[u + i - j, j] == a[i,j]

    where a is the matrix. This format is compatible with ``scipy.linalg.solve_banded``
    and ``scipy.linalg.eig_banded``.
    """
    pass


def matrix_hamiltonian(
    system: HamiltonianType,
    num_sites: int,
    params: dict[str, complex | Callable],
    hermitian: bool = True,
) -> np.ndarray:
    """Construct the matrix representation of the Hamiltonian.

    Mainly used for testing purposes.
    """
    # Initialize the Hamiltonian matrix
    dim = next(iter(system[0].values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)
    H = np.zeros((num_sites, num_sites, dim, dim), dtype=dtype)

    # Fill the Hamiltonian matrix
    for hop_length, terms in system.items():
        for term_name, term_matrix in terms.items():
            value = params[term_name]
            if callable(value):
                value = value(np.arange(num_sites - abs(hop_length)))
            else:
                value = np.full(num_sites - abs(hop_length), value)
            H += (
                np.diag(value, k=hop_length)[..., None, None]
                * term_matrix[None, None, ...]
            )
            if hop_length and hermitian:
                H += (
                    np.diag(value, k=-hop_length)[..., None, None]
                    * term_matrix.conj().T[None, None, ...]
                )

    # Reshape the Hamiltonian matrix to 2D
    return H.transpose(0, 2, 1, 3).reshape(num_sites * dim, num_sites * dim)
