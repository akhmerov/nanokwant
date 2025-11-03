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


def _to_banded(
    a: np.ndarray, *, shrink: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert a full 2D array to banded format.

    The diagonal ordered format is defined as

        ab[u + i - j, j] == a[i, j]

    Parameters
    ----------
    a : np.ndarray
        2D matrix to convert (shape (dim, dim)).
    shrink : bool, optional
        If True, trim zero-only rows from the top/bottom of the per-term
        band and also return the detected (l, u) bandwidth in row-offset
        units. When False (default) the full per-term band is returned.

    Returns
    -------
    (ab, (l, u)) : tuple
        Tuple of the banded matrix and the per-term lower/upper row offsets
        measured from the central diagonal row. If ``shrink`` is False the
        full per-term band is returned and the offsets are ``(n-1, n-1)``
        (i.e. the untrimmed base bandwidth). If ``shrink`` is True the
        returned band is trimmed of all-zero top/bottom rows and the
        offsets reflect the trimmed band.
    """
    (m, n) = a.shape
    # full band height (diagonals) for an n x n block is n + n - 1
    m_full = m + n - 1
    ab = np.zeros((m_full, n), dtype=a.dtype)
    for i in range(n):
        ab[n - 1 - i : m_full - i, i] = a[:, i]

    if not shrink:
        # Return the full per-term band and the base bandwidth offsets
        return ab, (n - 1, n - 1)

    # Trim zero-only rows from top/bottom.
    row_mask = np.any(ab != 0, axis=1)
    if not row_mask.any():
        # zero matrix -> return a minimal 1xN zero band and zero bandwidths
        return np.zeros((1, n), dtype=ab.dtype), (0, 0)

    first = int(np.argmax(row_mask))
    last = int(len(row_mask) - 1 - np.argmax(row_mask[::-1]))
    banded_trim = ab[first : last + 1]

    # Calculate per-term l/u measured in rows relative to central diagonal
    base_bandwidth = n - 1
    top_trim = first
    bottom_trim = len(row_mask) - 1 - last
    u_term = base_bandwidth - top_trim
    l_term = base_bandwidth - bottom_trim

    return banded_trim, (l_term, u_term)


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


def _prepare_param_arrays(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict[str, complex | Callable | np.ndarray],
) -> tuple[int, dict[tuple[int, str], np.ndarray]]:
    """Infer/validate num_sites and convert params to arrays keyed by (hop_length, term_name).
    Only non-negative hoppings are considered for inference/validation; hermitian expansion is separate."""
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
            raise ValueError(
                "num_sites could not be inferred: provide num_sites or array parameters."
            )
        if len(candidates) != 1:
            raise ValueError(
                f"Inconsistent array parameter lengths; inferred candidates {candidates} for num_sites."
            )
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
    param_arrays: dict[tuple[int, str], np.ndarray] = {}
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
    return num_sites, param_arrays


def _ensure_nonhermitian(
    system: HamiltonianType, param_arrays: dict[tuple[int, str], np.ndarray]
) -> tuple[HamiltonianType, dict[tuple[int, str], np.ndarray]]:
    """Return a non-Hermitian-expanded system and matching parameter arrays.
    Adds negative hoppings by conjugating matrices and parameter arrays.
    Real-valued parameter arrays are reused (not copied needlessly)."""
    new_system: HamiltonianType = {
        k: {n: m for n, m in terms.items()} for k, terms in system.items() if k >= 0
    }
    new_params = {(k, n): arr for (k, n), arr in param_arrays.items() if k >= 0}
    for hop_length, terms in list(system.items()):
        if hop_length > 0:
            neg = -hop_length
            if (
                neg in new_system
            ):  # skip if user provided explicitly (non-hermitian use-case)
                continue
            new_terms = {}
            for term_name, term_matrix in terms.items():
                new_terms[term_name] = term_matrix.conj().T
                arr = param_arrays[(hop_length, term_name)]
                if np.iscomplexobj(arr):
                    arr_neg = arr.conj()
                else:
                    arr_neg = arr  # reuse real array
                new_params[(neg, term_name)] = arr_neg
            new_system[neg] = new_terms
    return new_system, new_params


def _assemble_banded_matrix(
    system: HamiltonianType,
    num_sites: int,
    param_arrays: dict[tuple[int, str], np.ndarray],
    dim: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Assemble the banded matrix for the provided system definition."""
    if not system:
        raise ValueError("system must contain at least one hopping term")

    trimmed_system: dict[int, dict[str, tuple[np.ndarray, int, int]]] = {}
    min_delta = 0
    max_delta = 0

    for hop, terms in system.items():
        trimmed_terms: dict[str, tuple[np.ndarray, int, int]] = {}
        for name, matrix in terms.items():
            # Convert the small (dim x dim) term matrix to a trimmed per-term
            # band and obtain its per-term (l,u) measured in rows relative to
            # the central diagonal row.
            banded_trim, (l_term, u_term) = _to_banded(matrix)

            # A completely zero term will return a 1xN zero block with
            # l_term==u_term==0 — treat that as no contribution.
            if banded_trim.size == 0 or (
                banded_trim.shape[0] == 1 and not np.any(banded_trim)
            ):
                continue

            trimmed_terms[name] = (banded_trim, l_term, u_term)

            # Track global min/max deltas (in row-offset units) contributed by
            # this hopping length. hop*dim shifts the block vertically; we
            # subtract/add the term l/u to get absolute offsets.
            min_delta = min(min_delta, hop * dim - l_term)
            max_delta = max(max_delta, hop * dim + u_term)
        if trimmed_terms:
            trimmed_system[hop] = trimmed_terms

    if not trimmed_system:
        H = np.zeros((1, num_sites * dim), dtype=dtype)
        return H, (0, 0)

    # Ensure the global deltas at least include the central diagonal (0).
    # l_full and u_full are expressed in row-offset units from the central
    # diagonal row. For example, l_full is how many rows below the center we
    # need (lower bandwidth), u_full is how many rows above the center (upper).
    min_delta = min(min_delta, 0)
    max_delta = max(max_delta, 0)
    l_full = -min_delta
    u_full = max_delta

    H = np.zeros((l_full + u_full + 1, num_sites * dim), dtype=dtype)

    # Write each trimmed per-term band into the global band array H.
    # Loop over hopping lengths and the per-term trimmed blocks created above.
    for hop_length, terms in trimmed_system.items():
        # Columns affected by a hopping of length `hop_length` span sites
        # [term_start, term_end) in site-space; convert to flattened indices
        # by multiplying by `dim` later when slicing `H`.
        term_start = max(0, hop_length)
        term_end = num_sites + min(0, hop_length)

        for term_name, (banded_trim, l_term, u_term) in terms.items():
            # Parameter values for each site for this hopping+term. Shape (N-hop_length,)
            values = param_arrays[(hop_length, term_name)]

            # Compute the top row in the global H where this per-term trimmed
            # band should begin. `u_full` is the absolute row index of the
            # topmost global band row; subtracting the (hop shift + per-term
            # upper offset) aligns the per-term upper row correctly.
            row_start = u_full - (hop_length * dim + u_term)

            # Expand the per-term trimmed band into a shaped block that matches
            # (rows, cols_in_flattened_space). We multiply `values` into the
            # band rows first, then transpose/reshape so rows correspond to
            # diagonals and columns to flattened site*dim indices.
            term = (
                (values[..., None, None] * banded_trim[None, ...])
                .transpose(1, 0, 2)
                .reshape(banded_trim.shape[0], -1)
            )

            # Add the small block into H at the computed row/column slice.
            H[
                row_start : row_start + banded_trim.shape[0],
                term_start * dim : term_end * dim,
            ] += term

    return H, (l_full, u_full)


def hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict[str, complex | Callable | np.ndarray],
    hermitian: bool = True,
    format: str = "general",
) -> np.ndarray:
    """Generate the finite system Hamiltonian in band matrix format.

    The diagonal ordered format defined as

        ab[u + i - j, j] == a[i,j]

    where a is the matrix.

    Parameters
    ----------
    system : HamiltonianType
        Dictionary mapping hopping lengths to term dictionaries.
    num_sites : int | None
        Number of sites. Can be None if array parameters are provided.
    params : dict
        Parameter values (constants, callables, or arrays).
    hermitian : bool, optional
        If True, expand system to include negative hoppings. Default is True.
    format : str, optional
        Output format: "general" (default) for scipy.linalg.solve_banded,
        or "eig_banded" for scipy.linalg.eig_banded (only upper bands, hermitian only).

    Returns
    -------
    tuple[np.ndarray, tuple[int, int]]
        Banded matrix and (l, u) bandwidths. For "eig_banded" format, l is always 0
        and only the upper triangle is stored.
    """
    if format not in ("general", "eig_banded"):
        raise ValueError(f"format must be 'general' or 'eig_banded', got {format!r}")
    if format == "eig_banded" and not hermitian:
        raise ValueError("eig_banded format requires hermitian=True")
    if hermitian:
        if any(hop_length < 0 for hop_length in system):
            raise ValueError("Hermitian Hamiltonian cannot have negative hoppings.")
    # Prepare parameter arrays (no hermitian expansion here)
    num_sites, param_arrays = _prepare_param_arrays(system, num_sites, params)
    # Determine the block dimension `dim` from any term matrix available in
    # the provided system. Previously we assumed `system[0]` existed which
    # breaks for systems that have no onsite terms (hop=0). Use the first
    # available term mapping instead.
    try:
        first_terms = next(iter(system.values()))
    except StopIteration:
        raise ValueError("system must contain at least one hopping term")
    dim = next(iter(first_terms.values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)

    if format == "eig_banded":
        # For eig_banded we only need non-negative hoppings and only the
        # upper-triangular part of onsite (hop==0) term matrices. Applying
        # `np.triu` here avoids leaving lower-triangle content that would
        # otherwise be removed by slicing the assembled band later.
        # Build upper-only system: keep non-negative hops, triangularize onsite
        # matrices with `np.triu` and keep other hops unchanged.
        upper_system = {
            hop: {name: (mat if hop else np.triu(mat)) for name, mat in terms.items()}
            for hop, terms in system.items()
            if hop >= 0
        }

        upper_params = {
            key: values for key, values in param_arrays.items() if key[0] >= 0
        }

        ab_shrunk, (_, u_shrunk) = _assemble_banded_matrix(
            upper_system, num_sites, upper_params, dim, dtype
        )
        return ab_shrunk, (0, u_shrunk)

    if hermitian:
        system, param_arrays = _ensure_nonhermitian(system, param_arrays)
    return _assemble_banded_matrix(system, num_sites, param_arrays, dim, dtype)


def matrix_hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict[str, complex | Callable | np.ndarray],
    hermitian: bool = True,
) -> np.ndarray:
    """Construct the matrix representation of the Hamiltonian.

    Mainly used for testing purposes.
    """
    if hermitian:
        if any(hop_length < 0 for hop_length in system):
            raise ValueError("Hermitian Hamiltonian cannot have negative hoppings.")
    num_sites, param_arrays = _prepare_param_arrays(system, num_sites, params)
    if hermitian:
        system, param_arrays = _ensure_nonhermitian(system, param_arrays)
    try:
        first_terms = next(iter(system.values()))
    except StopIteration:
        raise ValueError("system must contain at least one hopping term")
    dim = next(iter(first_terms.values())).shape[0]
    dtype = _hamiltonian_dtype(system, params)
    H = np.zeros((num_sites, num_sites, dim, dim), dtype=dtype)
    for hop_length, terms in system.items():
        for term_name, term_matrix in terms.items():
            value = param_arrays[(hop_length, term_name)]
            H += (
                np.diag(value, k=hop_length)[..., None, None]
                * term_matrix[None, None, ...]
            )
    return H.transpose(0, 2, 1, 3).reshape(num_sites * dim, num_sites * dim)
