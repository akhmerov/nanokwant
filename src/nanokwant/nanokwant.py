import numpy as np
from typing import Callable

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


def _to_nonhermitian_format(system: HamiltonianType) -> HamiltonianType:
    """Remove negative hoppings and add conjugate transposes of positive hoppings."""
    return {
        hop_length: {term_name: term_matrix for term_name, term_matrix in terms.items()}
        for hop_length, terms in system.items()
        if hop_length >= 0
    } | {
        -hop_length: {
            term_name: term_matrix.conj().T for term_name, term_matrix in terms.items()
        }
        for hop_length, terms in system.items()
        if hop_length > 0
    }


def _shrink_banded(
    ab: np.ndarray,
    l: int,  # noqa: E741
    u: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Eliminate top and bottom zero rows from a banded matrix."""
    zero_rows = np.all(ab == 0, axis=1)
    # find the first and last non-zero rows
    first_nonzero = np.argmax(~zero_rows)
    last_nonzero = len(zero_rows) - np.argmax(~zero_rows[::-1])
    return ab[first_nonzero:last_nonzero], (l - last_nonzero, u - first_nonzero)


def hamiltonian(
    system: HamiltonianType,
    num_sites: int,
    params: dict[str, complex | Callable],
    hermitian: bool = True,
) -> np.ndarray:
    """Generate the finite system Hamiltonian in band matrix format.

    The diagonal ordered format defined as

        ab[u + i - j, j] == a[i,j]

    where a is the matrix. This format is compatible with ``scipy.linalg.solve_banded``
    and ``scipy.linalg.eig_banded``.
    """
    if hermitian:
        if any(hop_length < 0 for hop_length in system):
            raise ValueError("Hermitian Hamiltonian cannot have negative hoppings.")
        system = _to_nonhermitian_format(system)
    dim = next(iter(system[0].values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)

    # Convert all the matrices in system to banded format
    system = {
        hop_length: {
            term_name: _to_banded(term_matrix)
            for term_name, term_matrix in terms.items()
        }
        for hop_length, terms in system.items()
    }
    # Bandwidth of the Hamiltonian
    l, u = max(-k for k in system.keys()), max(system.keys())  # noqa: E741
    # Band matrix bandwidths. Each block extends it by dim, except for the diagonal, which
    # contributes dim - 1.
    l_full = dim * l + (dim - 1)
    u_full = dim * u + (dim - 1)

    hamiltonian_shape = (l_full + u_full + 1, num_sites * dim)
    H = np.zeros(hamiltonian_shape, dtype=dtype)

    for hop_length, terms in system.items():
        term_start, term_end = max(0, hop_length), num_sites + min(0, hop_length)
        term_bottom = H.shape[0] - (l + hop_length) * dim
        for term_name, term_matrix in terms.items():
            value = params[term_name]
            if callable(value):
                value = value(np.arange(num_sites - abs(hop_length)))
            else:
                value = np.full(num_sites - abs(hop_length), value)
            term = (
                (value[..., None, None] * term_matrix[None, ...])
                .transpose(1, 0, 2)
                .reshape(term_matrix.shape[0], -1)
            )
            H[
                term_bottom - term.shape[0] : term_bottom,
                term_start * dim : term_end * dim,
            ] += term

    return _shrink_banded(H, l_full, u_full)


def matrix_hamiltonian(
    system: HamiltonianType,
    num_sites: int,
    params: dict[str, complex | Callable],
    hermitian: bool = True,
) -> np.ndarray:
    """Construct the matrix representation of the Hamiltonian.

    Mainly used for testing purposes.
    """
    if hermitian:
        if any(hop_length < 0 for hop_length in system):
            raise ValueError("Hermitian Hamiltonian cannot have negative hoppings.")
        system = _to_nonhermitian_format(system)

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

    # Reshape the Hamiltonian matrix to 2D
    return H.transpose(0, 2, 1, 3).reshape(num_sites * dim, num_sites * dim)
