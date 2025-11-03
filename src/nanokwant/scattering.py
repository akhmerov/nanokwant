"""Scattering system construction for nanokwant.

This module provides functionality to construct scattering systems with leads
attached, using Kwant's modes() function for lead mode computation.
"""

import numpy as np
from typing import Literal
from kwant.physics import leads as kwant_leads
from .nanokwant import (
    HamiltonianType,
    hamiltonian,
    _hamiltonian_dtype,
    _prepare_param_arrays,
)


def _extract_lead_hamiltonians(
    system: HamiltonianType,
    param_arrays: dict[tuple[int, str], np.ndarray],
    site_idx: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract lead cell and hopping Hamiltonians from boundary site.

    Parameters
    ----------
    system : HamiltonianType
        System definition in Hermitian format (only non-negative hoppings).
    param_arrays : dict
        Parameter arrays keyed by (hop_length, term_name).
    site_idx : int
        Index of the boundary site (0 for left, num_sites-1 for right).
    dtype : np.dtype
        Data type for the matrices.

    Returns
    -------
    h_cell : np.ndarray
        Lead unit cell Hamiltonian.
    h_hop : np.ndarray
        Inter-cell hopping matrix.
    """
    # Get dimension from any term matrix
    first_terms = next(iter(system.values()))
    dim = next(iter(first_terms.values())).shape[0]

    # Build cell Hamiltonian from onsite terms
    h_cell = np.zeros((dim, dim), dtype=dtype)
    if 0 in system:
        for term_name, term_matrix in system[0].items():
            val = param_arrays[(0, term_name)][site_idx]
            h_cell += val * term_matrix

    # Build hopping Hamiltonian - system should only have hopping length 1
    # since we validated it's in Hermitian format
    h_hop = np.zeros((dim, dim), dtype=dtype)
    if 1 in system:
        for term_name, term_matrix in system[1].items():
            # For left lead (site_idx=0), use hopping from site 0
            # For right lead (site_idx=num_sites-1), use hopping from site num_sites-1
            # which is the last element in the array of length num_sites-1
            val = param_arrays[(1, term_name)][site_idx if site_idx == 0 else -1]
            h_hop += val * term_matrix

    return h_cell, h_hop


def _augment_with_lead_block(
    H_band: np.ndarray,
    l: int,
    u: int,
    coupling_col: np.ndarray,
    coupling_row: np.ndarray,
    diag_block: np.ndarray,
    iface_indices: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Augment banded Hamiltonian with a lead block.

    This converts the banded representation temporarily to dense, augments it,
    and converts back. This is acceptable because the augmented part is small
    (only the evanescent modes).

    Parameters
    ----------
    H_band : np.ndarray
        Current Hamiltonian in banded format.
    l, u : int
        Current lower and upper bandwidths.
    coupling_col : np.ndarray
        Column coupling block to interface (dense, shape n_current x n_modes).
    coupling_row : np.ndarray
        Row coupling block from interface (dense, shape n_modes x n_current).
    diag_block : np.ndarray
        Diagonal block for new modes (dense, shape n_modes x n_modes).
    iface_indices : np.ndarray
        Indices of interface sites in current system.

    Returns
    -------
    H_band_new : np.ndarray
        Augmented Hamiltonian in banded format.
    l_new, u_new : int
        New lower and upper bandwidths.
    """
    # Convert current banded to dense
    n_current = H_band.shape[1]
    H_dense = np.zeros((n_current, n_current), dtype=H_band.dtype)
    for i in range(n_current):
        for j in range(max(0, i - l), min(n_current, i + u + 1)):
            H_dense[i, j] = H_band[u + i - j, j]

    # Create augmented dense matrix
    n_modes = diag_block.shape[0]
    n_total = n_current + n_modes
    H_aug = np.zeros((n_total, n_total), dtype=H_dense.dtype)

    # Copy original Hamiltonian
    H_aug[:n_current, :n_current] = H_dense

    # Add coupling blocks at interface
    H_aug[iface_indices[:, None], n_current + np.arange(n_modes)] = coupling_col[
        iface_indices
    ]
    H_aug[n_current + np.arange(n_modes), iface_indices] = coupling_row[
        :, iface_indices
    ]

    # Add diagonal block
    H_aug[n_current:, n_current:] = diag_block

    # Convert back to banded
    from .nanokwant import _to_banded

    H_band_new, (l_new, u_new) = _to_banded(H_aug)

    return H_band_new, l_new, u_new


def scattering_system(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    energy: float,
    leads: Literal["both", "left", "right"] = "both",
) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray], list[np.ndarray], list[int]]:
    """Construct the scattering system in banded format.

    This function constructs a linear system for scattering calculations with
    leads attached to a 1D tight-binding system. The leads are defined using
    the first/last hopping and onsite terms of the system.

    Parameters
    ----------
    system : HamiltonianType
        Dictionary mapping hopping lengths to term dictionaries.
        Must be in Hermitian format (only non-negative hoppings).
        Must only contain nearest-neighbor hopping (hop lengths 0 and 1).
    num_sites : int | None
        Number of sites in the scattering region.
    params : dict
        Parameter values (constants, callables, or arrays).
    energy : float
        Energy at which to compute the scattering problem.
    leads : {"both", "left", "right"}
        Which leads to attach. "both" attaches leads at both ends,
        "left" only at the beginning, "right" only at the end.

    Returns
    -------
    lhs : np.ndarray
        Left-hand side matrix in banded format (diagonal-ordered).
    (l, u) : tuple of int
        Lower and upper bandwidth of the LHS matrix.
    rhs : list of np.ndarray
        List of right-hand side matrices, one per incoming lead.
        Each matrix has shape (total_size, num_propagating_modes).
    indices : list of np.ndarray
        List of arrays indicating which rows correspond to each lead's outgoing modes.
    nmodes_list : list of int
        List of number of propagating modes for each lead.

    Raises
    ------
    ValueError
        If the system contains hoppings longer than nearest-neighbor.
    ValueError
        If the system contains negative hoppings (not in Hermitian format).

    Notes
    -----
    The implementation follows Kwant's approach to constructing scattering systems
    (see kwant.solvers.common.SparseSolver._make_linear_sys), but constructs the
    result in banded matrix format for efficiency in 1D systems.

    The linear system constructed is similar to Kwant's:
    We augment the Hamiltonian with blocks for the lead modes, creating:

        [[H - E*I,  V†u_out],  [[ψ_scatter], = [[0        ],
         [V,        -λ_out^-1]] [ψ_lead   ]]    [-u_in λ_in^-1]]

    where:
    - H is the scattering region Hamiltonian
    - V is the coupling to the lead
    - u_out, u_in are outgoing/incoming mode wave functions
    - λ_out, λ_in are the outgoing/incoming mode eigenvalues
    """
    # Validate that system only has nearest-neighbor hopping
    max_hop = max(abs(hop) for hop in system.keys())
    if max_hop > 1:
        raise ValueError(
            f"System contains hopping length {max_hop}, but scattering "
            "systems only support nearest-neighbor hopping (max length 1)."
        )

    # Validate that system is in Hermitian format (no negative hoppings)
    if any(hop < 0 for hop in system.keys()):
        raise ValueError(
            "System must be in Hermitian format (no negative hoppings). "
            "Only provide hopping length 1, not -1."
        )

    # Validate that we have the necessary terms for leads
    if 1 not in system:
        raise ValueError(
            "System must contain nearest-neighbor hopping (length 1) for leads."
        )

    # Prepare parameter arrays and determine system size
    num_sites, param_arrays = _prepare_param_arrays(system, num_sites, params)

    # Get the dimension from any term matrix
    first_terms = next(iter(system.values()))
    dim = next(iter(first_terms.values())).shape[0]

    # Determine dtype - ensure it's complex to handle mode wave functions
    dtype = _hamiltonian_dtype(system, params)
    if not np.iscomplexobj(np.zeros(1, dtype=dtype)):
        dtype = np.result_type(dtype, np.complex128)

    # Construct the scattering region Hamiltonian in banded format
    # Use hermitian=True to expand to both directions
    H_band, (l, u) = hamiltonian(system, num_sites, params, hermitian=True)

    # Subtract energy from the diagonal
    # The diagonal is at row index u in the banded format
    H_band[u, :] -= energy

    # Determine which leads to process
    lead_configs = []
    if leads in ("both", "left"):
        lead_configs.append(("left", 0))
    if leads in ("both", "right"):
        lead_configs.append(("right", num_sites - 1))

    # Process each lead and augment the system
    indices_list = []
    nmodes_list = []
    lead_info_list = []
    current_size = num_sites * dim

    for lead_name, site_idx in lead_configs:
        # Extract lead Hamiltonians
        h_cell, h_hop = _extract_lead_hamiltonians(
            system, param_arrays, site_idx, dtype
        )

        # Compute modes using Kwant
        prop_modes, stab_modes = kwant_leads.modes(h_cell, h_hop)

        # Get the mode information
        u_modes = stab_modes.vecs  # All eigenvectors
        ulinv = stab_modes.vecslmbdainv  # u * lambda^-1
        nprop = stab_modes.nmodes  # Number of propagating modes
        svd_v = stab_modes.sqrt_hop  # sqrt of hopping matrix from SVD

        # Split into outgoing and incoming modes
        u_out = u_modes[:, nprop:]
        ulinv_out = ulinv[:, nprop:]
        u_in = u_modes[:, :nprop]
        ulinv_in = ulinv[:, :nprop]

        # Determine interface indices in the scattering region
        if lead_name == "left":
            iface_indices = np.arange(dim)
        else:  # right
            iface_indices = np.arange((num_sites - 1) * dim, num_sites * dim)

        # Construct the coupling matrices
        vdag = svd_v
        vdag_u_out = vdag @ u_out  # dim x n_out
        vdag_u_in = vdag @ u_in  # dim x nprop

        n_out = u_out.shape[1]

        # Augment the Hamiltonian with lead blocks
        if n_out > 0:
            # Create coupling blocks
            coupling_col = np.zeros((current_size, n_out), dtype=dtype)
            coupling_col[iface_indices, :] = vdag_u_out

            coupling_row = np.zeros((n_out, current_size), dtype=dtype)
            coupling_row[:, iface_indices] = vdag.T.conj()

            diag_block = -ulinv_out

            # Augment using helper function
            H_band, l, u = _augment_with_lead_block(
                H_band, l, u, coupling_col, coupling_row, diag_block, iface_indices
            )

            current_size += n_out

        # Store indices for this lead's outgoing modes
        indices_list.append(np.arange(current_size - n_out, current_size))
        nmodes_list.append(nprop)

        # Store information needed to construct RHS
        lead_info_list.append(
            {
                "iface_indices": iface_indices,
                "vdag_u_in": vdag_u_in,
                "ulinv_in": ulinv_in,
                "n_in": nprop,
                "n_out": n_out,
            }
        )

    # Construct RHS matrices for all leads
    rhs_list = []
    for lead_idx, info in enumerate(lead_info_list):
        if info["n_in"] > 0:
            rhs = np.zeros((current_size, info["n_in"]), dtype=dtype)
            rhs[info["iface_indices"], :] = -info["vdag_u_in"]
            # The ulinv_in contribution goes to the rows corresponding to this lead's outgoing modes
            if info["n_out"] > 0:
                rhs[indices_list[lead_idx], :] = info["ulinv_in"]
            rhs_list.append(rhs)
        else:
            rhs_list.append(np.zeros((current_size, 0), dtype=dtype))

    return H_band, (l, u), rhs_list, indices_list, nmodes_list


def compute_smatrix(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    energy: float,
    leads: Literal["both", "left", "right"] = "both",
) -> np.ndarray:
    """Compute the scattering matrix for a system with leads.

    Parameters
    ----------
    system : HamiltonianType
        System definition in Hermitian format.
    num_sites : int | None
        Number of sites in the scattering region.
    params : dict
        Parameter values.
    energy : float
        Energy at which to compute the S-matrix.
    leads : {"both", "left", "right"}
        Which leads to attach.

    Returns
    -------
    S : np.ndarray
        Scattering matrix with shape (total_modes, total_modes).
    """
    from scipy.linalg import solve_banded

    # Get the scattering system
    lhs, (l, u), rhs_list, indices_list, nmodes_list = scattering_system(
        system, num_sites, params, energy, leads
    )

    # Solve for each incoming lead
    solutions = []
    for rhs in rhs_list:
        if rhs.shape[1] > 0:
            sol = solve_banded((l, u), lhs, rhs)
            solutions.append(sol)
        else:
            solutions.append(np.zeros((lhs.shape[1], 0), dtype=lhs.dtype))

    # Extract S-matrix elements
    total_modes = sum(nmodes_list)
    S = np.zeros((total_modes, total_modes), dtype=complex)

    # Fill in the S-matrix
    col_offset = 0
    for j, (sol, nmodes_j) in enumerate(zip(solutions, nmodes_list)):
        row_offset = 0
        for i, (indices, nmodes_i) in enumerate(zip(indices_list, nmodes_list)):
            # Extract outgoing amplitudes
            S[
                row_offset : row_offset + nmodes_i, col_offset : col_offset + nmodes_j
            ] = sol[indices[:nmodes_i], :nmodes_j]
            row_offset += nmodes_i
        col_offset += nmodes_j

    return S
