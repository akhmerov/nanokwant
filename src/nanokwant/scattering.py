"""Scattering system construction for nanokwant.

This module provides functionality to construct scattering systems with leads
attached, using Kwant's modes() function for lead mode computation.
"""

import numpy as np
import scipy.sparse as sp
from typing import Literal
from kwant.physics import leads as kwant_leads
from .nanokwant import (
    HamiltonianType,
    hamiltonian,
    matrix_hamiltonian,
    _hamiltonian_dtype,
    _prepare_param_arrays,
    _to_banded,
)


def _dense_to_banded(A: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Convert a dense matrix to banded format, detecting the bandwidth.
    
    Parameters
    ----------
    A : np.ndarray
        Dense matrix to convert.
        
    Returns
    -------
    A_band : np.ndarray
        Matrix in banded format (diagonal-ordered).
    (l, u) : tuple of int
        Lower and upper bandwidth.
    """
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix must be square")
    
    # Find the actual bandwidth by looking at non-zero elements
    l = 0  # noqa: E741
    u = 0
    for i in range(n):
        for j in range(n):
            if abs(A[i, j]) > 1e-15:
                if i > j:
                    l = max(l, i - j)  # noqa: E741
                else:
                    u = max(u, j - i)
    
    # Create banded format
    A_band = np.zeros((l + u + 1, n), dtype=A.dtype)
    for i in range(n):
        for j in range(max(0, i - l), min(n, i + u + 1)):
            A_band[u + i - j, j] = A[i, j]
    
    return A_band, (l, u)


def scattering_system(
    system: HamiltonianType,
    num_sites: int | None,
    params: dict,
    energy: float,
    leads: Literal["both", "left", "right"] = "both",
) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray], list[np.ndarray]]:
    """Construct the scattering system in banded format.
    
    This function constructs a linear system for scattering calculations with
    leads attached to a 1D tight-binding system. The leads are defined using
    the first/last hopping and onsite terms of the system.
    
    Parameters
    ----------
    system : HamiltonianType
        Dictionary mapping hopping lengths to term dictionaries.
        Must only contain nearest-neighbor hopping (hop lengths 0 and ±1).
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
        
    Raises
    ------
    ValueError
        If the system contains hoppings longer than nearest-neighbor.
    ValueError
        If the system doesn't contain the required hopping terms for leads.
        
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
    
    # Validate that we have the necessary terms for leads
    if 0 not in system:
        raise ValueError("System must contain onsite terms (hopping length 0) for leads.")
    if 1 not in system and -1 not in system:
        raise ValueError(
            "System must contain nearest-neighbor hopping (length ±1) for leads."
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
    
    # Construct the scattering region Hamiltonian as a full matrix first
    # We'll convert to banded at the end after augmenting with lead blocks
    # Use hermitian=True to ensure we have both forward and backward hoppings
    H_matrix = matrix_hamiltonian(system, num_sites, params, hermitian=True)
    H_matrix = H_matrix - energy * np.eye(num_sites * dim, dtype=dtype)
    
    # Build lead Hamiltonians
    # For left lead, use parameters at site 0
    # For right lead, use parameters at last site
    h_cell_left = np.zeros((dim, dim), dtype=dtype)
    h_cell_right = np.zeros((dim, dim), dtype=dtype)
    
    for term_name, term_matrix in system[0].items():
        val_left = param_arrays[(0, term_name)][0]
        h_cell_left += val_left * term_matrix
        
        val_right = param_arrays[(0, term_name)][-1]
        h_cell_right += val_right * term_matrix
    
    # Extract inter-cell hopping for leads
    h_hop_left = np.zeros((dim, dim), dtype=dtype)
    h_hop_right = np.zeros((dim, dim), dtype=dtype)
    
    # For left lead: hopping from cell to the left into the system
    if 1 in system:
        for term_name, term_matrix in system[1].items():
            val = param_arrays[(1, term_name)][0]
            h_hop_left += val * term_matrix
    elif -1 in system:
        for term_name, term_matrix in system[-1].items():
            val = param_arrays[(-1, term_name)][0]
            h_hop_left += val * term_matrix.conj().T
    
    # For right lead: hopping from cell to the right into the system
    if -1 in system:
        for term_name, term_matrix in system[-1].items():
            val = param_arrays[(-1, term_name)][-1]
            h_hop_right += val * term_matrix
    elif 1 in system:
        for term_name, term_matrix in system[1].items():
            val = param_arrays[(1, term_name)][-1]
            h_hop_right += val * term_matrix.conj().T
    
    # Process leads and build the augmented system
    indices_list = []
    current_size = num_sites * dim
    
    # Determine which leads to process
    lead_configs = []
    if leads in ("both", "left"):
        lead_configs.append(("left", 0, h_cell_left, h_hop_left))
    if leads in ("both", "right"):
        lead_configs.append(("right", num_sites - 1, h_cell_right, h_hop_right))
    
    # Store lead information for constructing RHS later
    lead_info = []
    
    # Process each lead and augment the LHS
    for lead_name, interface_site, h_cell, h_hop in lead_configs:
        # Compute modes using Kwant
        prop_modes, stab_modes = kwant_leads.modes(h_cell, h_hop)
        
        # Get the mode information
        u = stab_modes.vecs  # All eigenvectors
        ulinv = stab_modes.vecslmbdainv  # u * lambda^-1
        nprop = stab_modes.nmodes  # Number of propagating modes
        svd_v = stab_modes.sqrt_hop  # sqrt of hopping matrix from SVD
        
        if len(u) == 0:
            # No modes, skip this lead
            lead_info.append(None)
            continue
        
        # Split into outgoing and incoming modes
        u_out = u[:, nprop:]
        ulinv_out = ulinv[:, nprop:]
        u_in = u[:, :nprop]
        ulinv_in = ulinv[:, :nprop]
        
        # Determine interface indices in the scattering region
        if lead_name == "left":
            iface_indices = np.arange(dim)
        else:  # right
            iface_indices = np.arange((num_sites - 1) * dim, num_sites * dim)
        
        # Construct the coupling matrix V† = svd_v
        # and its products with the mode vectors
        vdag = svd_v
        vdag_u_out = vdag @ u_out  # dim x n_out
        vdag_u_in = vdag @ u_in    # dim x nprop
        
        # Augment the Hamiltonian with lead blocks
        # Add columns for outgoing modes: V†u_out
        n_out = u_out.shape[1]
        n_in = nprop
        
        if n_out > 0:
            # Create a column block
            col_block = np.zeros((current_size, n_out), dtype=dtype)
            col_block[iface_indices, :] = vdag_u_out
            
            # Create a row block and diagonal block
            row_block = np.zeros((n_out, current_size), dtype=dtype)
            row_block[:, iface_indices] = vdag.T.conj()
            
            diag_block = -ulinv_out
            
            # Augment H_matrix
            H_matrix = np.block([
                [H_matrix, col_block],
                [row_block, diag_block]
            ])
            
            current_size += n_out
        
        # Store indices for this lead's outgoing modes
        indices_list.append(np.arange(current_size - n_out, current_size))
        
        # Store information needed to construct RHS
        lead_info.append({
            'iface_indices': iface_indices,
            'vdag_u_in': vdag_u_in,
            'ulinv_in': ulinv_in,
            'n_in': n_in,
            'n_out': n_out,
        })
    
    # Now construct RHS matrices for all leads
    # The RHS must match the final augmented size
    rhs_list = []
    for lead_idx, info in enumerate(lead_info):
        if info is None:
            rhs_list.append(None)
        elif info['n_in'] > 0:
            rhs = np.zeros((current_size, info['n_in']), dtype=dtype)
            rhs[info['iface_indices'], :] = -info['vdag_u_in']
            # The ulinv_in contribution goes to the rows corresponding to this lead's outgoing modes
            if info['n_out'] > 0:
                rhs[indices_list[lead_idx], :] = info['ulinv_in']
            rhs_list.append(rhs)
        else:
            rhs_list.append(None)
    
    # Convert the augmented matrix to banded format
    lhs_band, (l, u) = _dense_to_banded(H_matrix)
    
    return lhs_band, (l, u), rhs_list, indices_list
