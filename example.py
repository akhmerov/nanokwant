# %%
from importlib import reload
import timeit

import numpy as np
from scipy.linalg import eigvals_banded

import nanokwant.nanokwant
from nanokwant.nanokwant import hamiltonian, matrix_hamiltonian
import kwant

reload(nanokwant.nanokwant)

system = {
    0: {
        "mu": np.eye(2),
        "Ez": np.array([[0, 1], [1, 0]]),
    },
    1: {
        "t": np.eye(2),
    },
}
num_sites = 3
params = {
    "mu": 2.0,
    "Ez": lambda x: np.sin(x),
    "t": 1.0,
}
H, (l, u) = hamiltonian(system, num_sites, params)  # noqa: E741
H_matrix = matrix_hamiltonian(system, num_sites, params)

# %%
num_sites = 1000


# Time the hamiltonian generation
def time_hamiltonian():
    return hamiltonian(system, num_sites, params)


hamiltonian_time = timeit.timeit(time_hamiltonian, number=1)
print(f"Hamiltonian generation time: {hamiltonian_time:.4f} seconds")

# Get the hamiltonian for eigenvalue calculation
H, (l, u) = hamiltonian(system, num_sites, params)  # noqa: E741


# Time the eigenvalue calculation
def time_eigvals():
    return eigvals_banded(H[: u + 1], select="v", select_range=(-0.5, 0.6))


eigvals_time = timeit.timeit(time_eigvals, number=1)
print(f"Eigenvalue calculation time: {eigvals_time:.4f} seconds")
# %%
# %%


# Define the system in Kwant
def make_system(num_sites, params):
    def onsite(site, mu, Ez_func):
        x = float(site.pos[0])
        return mu * np.eye(2) + Ez_func(x) * np.array([[0, 1], [1, 0]])

    def hopping(site1, site2, t):
        return t * np.eye(2)

    # Create an empty 1D system
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()

    # Add sites and hoppings
    syst[(lat(i) for i in range(num_sites))] = lambda site: onsite(
        site, params["mu"], params["Ez"]
    )
    syst[kwant.builder.HoppingKind((1,), lat, lat)] = lambda site1, site2: hopping(
        site1, site2, params["t"]
    )

    # Finalize the system
    return syst.finalized()


# Create the same system with Kwant and time it
def time_kwant_system():
    return make_system(num_sites, params)


kwant_system_time = timeit.timeit(time_kwant_system, number=1)
print(f"Kwant system creation time: {kwant_system_time:.4f} seconds")

# Get the kwant system for further timing
kwant_system = make_system(num_sites, params)


# Time Kwant hamiltonian generation
def time_kwant_hamiltonian():
    return kwant_system.hamiltonian_submatrix(sparse=False)


kwant_hamiltonian_time = timeit.timeit(time_kwant_hamiltonian, number=1)
print(f"Kwant hamiltonian generation time: {kwant_hamiltonian_time:.4f} seconds")

# Get the hamiltonian for eigenvalue calculation
kwant_hamiltonian = kwant_system.hamiltonian_submatrix(sparse=False)


# Time Kwant eigenvalue calculation
def time_kwant_eigvals():
    return np.linalg.eigvals(kwant_hamiltonian)


kwant_eigvals_time = timeit.timeit(time_kwant_eigvals, number=1)
print(f"Kwant eigenvalue calculation time: {kwant_eigvals_time:.4f} seconds")
# %%
