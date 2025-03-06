import numpy as np
from nanokwant import matrix_hamiltonian

def test_matrix_hamiltonian():
    system = {
        0: {
            "mu": np.eye(2),
            "Ez": np.diag([1, -1]),
        },
        1: {
            "t": np.eye(2),
        }
    }
    num_sites = 3
    params = {
        "mu": 1.0,
        "Ez": 1.0,
        "t": 1.0,}
    expected_H = np.array([
        [2, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [1, 0, 2, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 2, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.complex128)
    H = matrix_hamiltonian(system, num_sites, params)
    np.testing.assert_equal(H, expected_H)


test_matrix_hamiltonian()
