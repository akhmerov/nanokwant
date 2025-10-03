# GitHub Copilot Instructions for nanokwant

## Project Overview

nanokwant is a lightweight, high-performance Python library for solving 1D tight-binding quantum systems. It provides a simple alternative to Kwant for cases where performance is critical and the system is one-dimensional.

## Key Concepts

### System Definition

A tight-binding system is defined as a dictionary where:
- Keys are hopping lengths (integers): `0` for onsite terms, `1` for nearest-neighbor hoppings, etc.
- Values are dictionaries mapping term names to matrix operators (numpy arrays)

Example:
```python
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
- **Arrays**: Explicit arrays specifying parameter values. Consistency rules when arrays are used:
  * Onsite (hopping length 0) arrays have length N (num_sites)
  * Hopping length k>0 arrays have length N-k
  * If any array parameter is provided, `num_sites` becomes optional and is inferred. All arrays must be mutually consistent or a ValueError is raised.

### Hamiltonian Generation

The library generates Hamiltonians in two formats:
1. **Banded format** (`hamiltonian`): Efficient diagonal-ordered format for use with `scipy.linalg.eigvals_banded` or `scipy.linalg.solve_banded`
2. **Matrix format** (`matrix_hamiltonian`): Full matrix representation, mainly for testing

## Code Style Guidelines

### General Principles
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Prefer numpy vectorized operations over loops
- Keep functions focused and modular

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private functions: prefix with underscore (e.g., `_hamiltonian_dtype`)

### Numpy and Scientific Computing
- Use numpy operations instead of Python loops when possible
- Prefer `np.ndarray` type hints over generic types
- Use descriptive variable names for array dimensions (e.g., `num_sites`, `dim`)
- For matrix operations, clearly document the expected shapes and formats

### Type Hints
- Use modern type hint syntax: `dict[str, int]` instead of `Dict[str, int]`
- For complex types, define type aliases (e.g., `HamiltonianType = dict[int, dict[str, np.ndarray]]`)
- Use `Callable` from typing for function parameters

### Documentation
- Use docstrings for all public functions
- Include mathematical formulas and format specifications where relevant
- Reference scipy functions when the output format is compatible

### Testing
- Use pytest for all tests
- Test both constant and function-valued parameters
- Verify consistency between banded and matrix formats
- Use `np.testing.assert_allclose` for floating-point comparisons
- Use `np.testing.assert_equal` for exact comparisons

## Performance Considerations

- The library is designed for performance; avoid unnecessary allocations
- Banded format is preferred over full matrices for large systems
- When working with eigenvalue problems, use `scipy.linalg.eigvals_banded` with the `select` parameter to compute only needed eigenvalues

## Development Workflow

- This project uses `pixi` for package management
- Run tests with: `pixi run pytest`
- Use pre-commit hooks: configured in `.pre-commit-config.yaml`
- Linting with ruff: `pixi run ruff`

## Common Patterns

### Adding a New Term Type
When adding support for new physical terms:
1. Add the term to the system dictionary with appropriate hopping length
2. Ensure the term matrix has the correct dimensions
3. Add corresponding parameter in the `params` dictionary
4. Test with both constant and function-valued parameters

### Working with Hermitian Systems
- By default, systems are assumed Hermitian
- Only specify positive hopping lengths; negative hoppings are automatically generated
- To work with non-Hermitian systems, set `hermitian=False`

### Debugging
- Use `matrix_hamiltonian` to verify results in full matrix form
- Compare eigenvalues between banded and matrix formats
- Check that the bandwidth is correctly computed

## Related Projects

- **Kwant**: Full-featured quantum transport library (this is a lightweight alternative)
- **scipy.linalg**: Used for efficient banded matrix operations
- **numpy**: Core array operations

## Notes

- Variable `l` (lowercase L) is used for lower bandwidth (allowed despite E741 noqa)
- The banded format follows scipy's diagonal-ordered convention: `ab[u + i - j, j] == a[i, j]`
- Bandwidth calculation accounts for block structure: each hopping extends bandwidth by `dim` (matrix dimension)
