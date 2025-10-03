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


def _prepare_params(system: HamiltonianType,
                     num_sites: int | None,
                     params: dict[str, complex | Callable | np.ndarray],
                     hermitian: bool) -> tuple[HamiltonianType, int, dict[tuple[int,str], np.ndarray]]:
    """Return (possibly expanded) system, resolved num_sites, and arrays for each (hop_length, term_name).

    All parameter specifications are converted to 1D numpy arrays of length num_sites - abs(hop_length).
    """
    if hermitian:
        if any(hop_length < 0 for hop_length in system):
            raise ValueError("Hermitian Hamiltonian cannot have negative hoppings.")
        system = _to_nonhermitian_format(system)

    # Infer num_sites
    if num_sites is None:
        candidates: set[int] = set()
        for hop_length, terms in system.items():
            if hop_length < 0:
                continue
            for term_name in terms:
                value = params[term_name]
                if callable(value):
                    continue
                arr = np.asarray(value)
                if arr.ndim == 0:
                    continue
                candidates.add(arr.shape[0] + (hop_length if hop_length > 0 else 0))
        if not candidates:
            raise ValueError("num_sites could not be inferred: provide num_sites or array parameters.")
        if len(candidates) != 1:
            raise ValueError(f"Inconsistent array parameter lengths; inferred candidates {candidates} for num_sites.")
        num_sites = candidates.pop()

    # Validate arrays
    for hop_length, terms in system.items():
        if hop_length < 0:
            continue
        for term_name in terms:
            value = params[term_name]
            if callable(value):
                continue
            arr = np.asarray(value)
            if arr.ndim == 0:
                continue
            expected = num_sites - (hop_length if hop_length > 0 else 0)
            if arr.shape[0] != expected:
                raise ValueError(
                    f"Parameter '{term_name}' for hopping length {hop_length} has length {arr.shape[0]}, expected {expected} for num_sites={num_sites}."
                )

    # Materialize arrays
    param_arrays: dict[tuple[int,str], np.ndarray] = {}
    for hop_length, terms in system.items():
        length = num_sites - abs(hop_length)
        xs = np.arange(length)
        for term_name in terms:
            value = params[term_name]
            if callable(value):
                arr = np.asarray(value(xs))
            else:
                raw = np.asarray(value)
                if raw.ndim == 0:
                    arr = np.full(length, raw)
                else:
                    arr = raw
            param_arrays[(hop_length, term_name)] = arr
    return system, num_sites, param_arrays


def hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict[str, complex | Callable | np.ndarray],
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
    # Prepare parameters (also expands hermitian system inside helper if needed)
    system, num_sites, param_arrays = _prepare_params(system, num_sites, params, hermitian)
    dim = next(iter(system[0].values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)
    # Convert matrices to banded once
    banded_system = {h: {n: _to_banded(m) for n,m in terms.items()} for h,terms in system.items()}
    l, u = max(-k for k in banded_system.keys()), max(banded_system.keys())  # noqa: E741
    l_full = dim * l + (dim - 1)
    u_full = dim * u + (dim - 1)
    H = np.zeros((l_full + u_full + 1, num_sites * dim), dtype=dtype)
    for hop_length, terms in banded_system.items():
        term_start, term_end = max(0, hop_length), num_sites + min(0, hop_length)
        term_bottom = H.shape[0] - (l + hop_length) * dim
        for term_name, term_matrix in terms.items():
            value = param_arrays[(hop_length, term_name)]
            term = (value[..., None, None] * term_matrix[None, ...]).transpose(1,0,2).reshape(term_matrix.shape[0], -1)
            H[term_bottom - term.shape[0]:term_bottom, term_start * dim: term_end * dim] += term
    return _shrink_banded(H, l_full, u_full)


def matrix_hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict[str, complex | Callable | np.ndarray],
    hermitian: bool = True,
) -> np.ndarray:
    """Construct the matrix representation of the Hamiltonian.

    Mainly used for testing purposes.
    """
    # Prepare parameters (handles hermitian processing too)
    system, num_sites, param_arrays = _prepare_params(system, num_sites, params, hermitian)
    dim = next(iter(system[0].values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)
    H = np.zeros((num_sites, num_sites, dim, dim), dtype=dtype)
    for hop_length, terms in system.items():
        for term_name, term_matrix in terms.items():
            value = param_arrays[(hop_length, term_name)]
            H += (np.diag(value, k=hop_length)[..., None, None] * term_matrix[None, None, ...])
    return H.transpose(0,2,1,3).reshape(num_sites * dim, num_sites * dim)
