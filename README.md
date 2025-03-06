# A simple 1D tight-binding solver

Sometimes your tight-binding system is just that simple that you don't need
Kwant but want it to be fast. That's when you use this.

A system is defined in the following way:

```python
system = {
    0: {
        "mu": np.eye(2),
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
hamiltonian = nanokwant.hamiltonian(system, num_sites, **params)
```

where `num_sites` is the number of sites in your system and `params` contains all parameters specified either as constant values or as functions of the site index.

## Developing

This project uses `pixi`. To run tests use `pixi run pytest`.
