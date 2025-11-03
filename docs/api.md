---
execute: true
---
# API Reference

## Core Functions

### `hamiltonian`

Generate the finite system Hamiltonian in band matrix format.

```python
hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    hermitian: bool = True,
    format: str = "general",
) -> tuple[np.ndarray, tuple[int, int]]
```

**Parameters:**

- `system`: Dictionary mapping hopping lengths to term dictionaries
- `num_sites`: Number of sites (can be `None` if array parameters are provided)
- `params`: Parameter values (constants, callables, or arrays)
- `hermitian`: If `True`, expand system to include negative hoppings
- `format`: Output format (`"general"` or `"eig_banded"`)

**Returns:**

- `H`: Banded matrix in diagonal-ordered format
- `(l, u)`: Lower and upper bandwidth

### `matrix_hamiltonian`

Construct the matrix representation of the Hamiltonian (mainly for testing).

```python
matrix_hamiltonian(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    hermitian: bool = True,
) -> np.ndarray
```

## Scattering Functions

### `scattering_system`

Construct the scattering system in banded format.

```python
scattering_system(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    energy: float,
    leads: Literal["both", "left", "right"] = "both",
) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray], list[np.ndarray], list[int]]
```

**Parameters:**

- `system`: System definition in Hermitian format (only non-negative hoppings)
- `num_sites`: Number of sites in the scattering region
- `params`: Parameter values
- `energy`: Energy at which to compute the scattering problem
- `leads`: Which leads to attach

**Returns:**

- `lhs`: Left-hand side matrix in banded format
- `(l, u)`: Lower and upper bandwidth
- `rhs`: List of right-hand side matrices (one per incoming lead)
- `indices`: List of arrays indicating rows for each lead's outgoing modes
- `nmodes`: List of number of propagating modes for each lead

### `compute_smatrix`

Compute the scattering matrix for a system with leads.

```python
compute_smatrix(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    energy: float,
    leads: Literal["both", "left", "right"] = "both",
) -> np.ndarray
```

**Parameters:**

- Same as `scattering_system`

**Returns:**

- `S`: Scattering matrix with shape `(total_modes, total_modes)`

## Type Definitions

### `HamiltonianType`

```python
HamiltonianType = dict[int, dict[str, np.ndarray]]
```

A dictionary mapping hopping lengths (int) to term dictionaries, where each term dictionary maps term names (str) to matrix operators (np.ndarray).
