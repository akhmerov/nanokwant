# Getting Started

## Installation

Install nanokwant using pip:

```bash
pip install nanokwant
```

Or use pixi for a complete environment:

```bash
pixi add nanokwant
```

## Basic Concepts

### System Definition

A tight-binding system is defined as a dictionary where:

- Keys are hopping lengths (integers): `0` for onsite terms, `1` for nearest-neighbor hoppings, etc.
- Values are dictionaries mapping term names to matrix operators (numpy arrays)

Example:

```python
import numpy as np

system = {
    0: {  # Onsite terms
        "mu": np.eye(2),           # Chemical potential
        "Ez": np.array([[0, 1], [1, 0]]),  # Zeeman term
    },
    1: {  # Nearest-neighbor hopping
        "t": np.eye(2),            # Hopping matrix
    }
}
```

### Parameters

Parameters can be:

- **Constants**: Simple numeric values (e.g., `"mu": 2.0`)
- **Functions**: Callables that take a site index array and return an array (e.g., `"Ez": lambda x: np.sin(x)`)
- **Arrays**: Explicit arrays specifying parameter values

### Hamiltonian Generation

```python
from nanokwant import hamiltonian

# Define parameters
params = {
    "mu": 2.0,
    "Ez": lambda x: np.sin(x),
    "t": 1.0,
}

# Generate Hamiltonian
H, (l, u) = hamiltonian(system, num_sites=100, params=params)
```

The Hamiltonian is returned in banded format for efficient storage and computation.

## Next Steps

- Learn about [Scattering Systems](scattering.md)
- Explore the [API Reference](../api.md)
