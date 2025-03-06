import numpy as np
from typing import Union, Callable

HamiltonianType = dict[int, dict[str, np.ndarray]]


def hamiltonian(
    system: HamiltonianType,
    num_sites: int,
    params: Union[complex, Callable],
    hermitian: bool = True,
) -> np.ndarray:
    """Generate the finite system Hamiltonian in band matrix format."""
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
    # TODO: determine type by using common type of all term matrices and all values
    H = np.zeros((num_sites, num_sites, dim, dim), dtype=np.complex128)

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
