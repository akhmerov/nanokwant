# A simple 1D tight-binding solver

Sometimes your tight-binding system is just that simple that you don't need
Kwant but want it to be fast. That's when you use this.

A system is defined in the following way:

```python
system = {
    0: {
        "mu": np.eye(2),
        "t": -np.eye(2),
        "Ez": np.diag([1, -1]),
    }
    1: {
        "t": np.eye(2),
    }
    hopping_length: {
        "term_name": term_matrix,
        ...
    }
}
```

Once you've defined your system, generate your Hamiltonian in a banded format using

```python
# To construct a system for solving linear equations:
H, (l, u) = nanokwant.hamiltonian(system, num_sites, params)
# For eigenvalue problems with hermitian matrices, use optimized format:
H_eig, (l, u) = nanokwant.hamiltonian(system, num_sites, params, format="eig_banded")
eigvals = scipy.linalg.eig_banded(H_eig, lower=False)
```

If all parameter values are constants or callables, you must specify `num_sites`.

Parameter values may also be numpy arrays. In that case `num_sites` becomes optional and will be
inferred from the array lengths under the consistency rules:

- For hopping length 0 (onsite), parameter arrays must have length N = num_sites
- For hopping length k>0, parameter arrays must have length N - k
All array-typed parameters must be mutually consistent; otherwise a ValueError is raised.

## Developing

This project uses `pixi`. To run tests use `pixi run pytest`.
