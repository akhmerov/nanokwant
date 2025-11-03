---
execute: true
---
# Scattering Systems

Nanokwant supports scattering matrix calculations for 1D systems with attached leads.

## Basic Usage

```python
import numpy as np
from nanokwant import scattering_system, compute_smatrix

# Define your system
system = {
    0: {"mu": np.eye(2)},
    1: {"t": np.eye(2)},
}

params = {"mu": 0.5, "t": 1.0}
num_sites = 50
energy = 0.5

# Compute the scattering matrix
S = compute_smatrix(system, num_sites, params, energy, leads="both")
```

## Lead Configuration

You can attach leads at different positions:

- `leads="both"`: Attach leads at both ends (default)
- `leads="left"`: Attach lead only at the left end
- `leads="right"`: Attach lead only at the right end

```python
# Single lead example
S_left = compute_smatrix(system, num_sites, params, energy, leads="left")
```

## Advanced: Direct Access to Linear System

For more control, you can access the linear system directly:

```python
from scipy.linalg import solve_banded

# Get the linear system
lhs, (l, u), rhs_list, indices_list, nmodes_list = scattering_system(
    system, num_sites, params, energy, leads="both"
)

# Solve for each incoming lead
for rhs, indices, nmodes in zip(rhs_list, indices_list, nmodes_list):
    solution = solve_banded((l, u), lhs, rhs)
    outgoing_amplitudes = solution[indices[:nmodes], :nmodes]
```

## Position-Dependent Parameters

Scattering systems support position-dependent parameters:

```python
def barrier(x):
    """Potential barrier in the middle."""
    return np.where((x >= 20) & (x <= 30), 2.0, 0.5)

params = {"mu": barrier, "t": 1.0}
S = compute_smatrix(system, num_sites, params, energy, leads="both")
```

## Evanescent Modes

The implementation properly handles evanescent modes using Kwant's stabilized mode computation:

```python
# System with different bandwidths for different orbitals
system = {
    0: {"mu": np.eye(2)},
    1: {"t": np.diag([1.0, 0.1])},  # Different hopping strengths
}

# At certain energies, some modes will be evanescent
S = compute_smatrix(system, num_sites, params, energy=0.3, leads="both")
```

## Validation

The scattering matrix should be unitary for Hermitian systems:

```python
# Check unitarity
unitarity_error = np.max(np.abs(S @ S.conj().T - np.eye(S.shape[0])))
print(f"Unitarity error: {unitarity_error:.2e}")
```
