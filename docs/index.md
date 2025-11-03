# nanokwant

A lightweight, high-performance Python library for solving 1D tight-binding quantum systems with support for scattering calculations.

## Features

- **Efficient banded matrix format**: Optimized for 1D systems
- **Scattering matrix calculations**: Compute transport properties with attached leads
- **Kwant integration**: Uses Kwant's robust mode computation for leads
- **Simple interface**: Easy-to-use Python API

## Installation

```bash
pip install nanokwant
```

Or with pixi:

```bash
pixi add nanokwant
```

## Quick Start

### Basic Hamiltonian

```python
import numpy as np
from nanokwant import hamiltonian

# Define a 1D tight-binding system
system = {
    0: {"mu": np.eye(2)},  # Onsite terms
    1: {"t": np.eye(2)},   # Nearest-neighbor hopping
}

params = {"mu": 0.5, "t": 1.0}
num_sites = 100

# Generate the Hamiltonian in banded format
H, (l, u) = hamiltonian(system, num_sites, params)
```

### Scattering Calculations

```python
from nanokwant import compute_smatrix

# Compute the scattering matrix
S = compute_smatrix(system, num_sites, params, energy=0.5, leads="both")

# S is unitary for Hermitian systems
print(f"Unitarity check: {np.allclose(S @ S.conj().T, np.eye(S.shape[0]))}")
```

## Documentation

- [Getting Started](user-guide/getting-started.md)
- [Scattering Systems](user-guide/scattering.md)
- [API Reference](api.md)

## License

[License information]
