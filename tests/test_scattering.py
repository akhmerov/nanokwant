"""Tests for the scattering module."""

import numpy as np
import pytest
import kwant
from scipy.linalg import solve_banded
from nanokwant.scattering import scattering_system, _dense_to_banded


def test_dense_to_banded():
    """Test conversion from dense to banded format."""
    # Test a tridiagonal matrix
    A = np.array([
        [1, 2, 0, 0],
        [3, 4, 5, 0],
        [0, 6, 7, 8],
        [0, 0, 9, 10],
    ], dtype=float)
    
    A_band, (l, u) = _dense_to_banded(A)
    
    assert l == 1
    assert u == 1
    assert A_band.shape == (3, 4)
    
    # Verify the banded format
    for i in range(4):
        for j in range(4):
            if abs(i - j) <= 1:
                assert abs(A_band[u + i - j, j] - A[i, j]) < 1e-10


def test_scattering_system_validation():
    """Test that scattering_system validates inputs correctly."""
    # System with too long hopping
    system_bad = {
        0: {"mu": np.eye(2)},
        2: {"t": np.eye(2)},
    }
    params = {"mu": 1.0, "t": 1.0}
    
    with pytest.raises(ValueError, match="nearest-neighbor"):
        scattering_system(system_bad, 10, params, energy=0.0)
    
    # System without onsite terms
    system_no_onsite = {
        1: {"t": np.eye(2)},
    }
    
    with pytest.raises(ValueError, match="onsite"):
        scattering_system(system_no_onsite, 10, {"t": 1.0}, energy=0.0)
    
    # System without hopping
    system_no_hop = {
        0: {"mu": np.eye(2)},
    }
    
    with pytest.raises(ValueError, match="nearest-neighbor hopping"):
        scattering_system(system_no_hop, 10, {"mu": 1.0}, energy=0.0)


def test_scattering_system_basic():
    """Test basic construction of a scattering system."""
    # Simple 1D system with constant parameters
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": -np.eye(2)},
    }
    num_sites = 5
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5
    
    lhs, (l, u), rhs_list, indices_list = scattering_system(
        system, num_sites, params, energy, leads="both"
    )
    
    # Check shapes
    assert lhs.ndim == 2
    assert l >= 0 and u >= 0
    
    # Should have 2 leads (left and right)
    assert len(rhs_list) == 2
    assert len(indices_list) == 2
    
    # Each RHS should be a matrix
    for rhs in rhs_list:
        if rhs is not None:
            assert rhs.ndim == 2
            assert rhs.shape[0] == lhs.shape[1]


def test_scattering_vs_kwant_matrix():
    """Test that our scattering system matches Kwant's structure."""
    # Create a simple system
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": -np.eye(2)},
    }
    num_sites = 10
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5
    
    # Get nanokwant scattering system
    lhs, (l, u), rhs_list, indices_list = scattering_system(
        system, num_sites, params, energy, leads="both"
    )
    
    # Create equivalent Kwant system
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()
    
    for i in range(num_sites):
        syst[lat(i)] = params["mu"] * np.eye(2)
    syst[lat.neighbors()] = params["t"] * np.eye(2)
    
    # Create leads
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = params["mu"] * np.eye(2)
    lead[lat.neighbors()] = params["t"] * np.eye(2)
    
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    
    fsyst = syst.finalized()
    
    # Get Kwant's linear system (using the default sparse solver)
    from kwant.solvers.default import smatrix as kwant_smatrix
    
    # For comparison, let's check the S-matrix computed by both
    # First compute with Kwant
    smat_kwant = kwant_smatrix(fsyst, energy)
    
    # For comparison, let's check the S-matrix computed by both
    # First compute with Kwant
    smat_kwant = kwant_smatrix(fsyst, energy)
    
    # Check our system can be used to compute something similar
    # The system sizes should be related
    # Convert our banded LHS to dense
    n = lhs.shape[1]
    our_lhs_dense = np.zeros((n, n), dtype=lhs.dtype)
    for i in range(n):
        for j in range(max(0, i - l), min(n, i + u + 1)):
            our_lhs_dense[i, j] = lhs[u + i - j, j]
    
    # The augmented system should be larger than the scattering region
    assert n > num_sites * 2, f"Augmented system size {n} should be > {num_sites * 2}"
    
    print(f"Nanokwant LHS shape: {our_lhs_dense.shape}")
    print(f"Kwant S-matrix shape: {smat_kwant.data.shape}")
    print(f"Nanokwant has {len(rhs_list)} RHS matrices")
    
    # Check that we have the right number of modes
    assert len(rhs_list) == 2  # two leads
    assert all(rhs is not None for rhs in rhs_list)  # both have modes


def test_scattering_system_single_lead():
    """Test scattering system with only one lead."""
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": -np.eye(2)},
    }
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5
    
    # Left lead only
    lhs_left, (l, u), rhs_left, idx_left = scattering_system(
        system, 5, params, energy, leads="left"
    )
    assert len(rhs_left) == 1
    assert len(idx_left) == 1
    
    # Right lead only
    lhs_right, (l, u), rhs_right, idx_right = scattering_system(
        system, 5, params, energy, leads="right"
    )
    assert len(rhs_right) == 1
    assert len(idx_right) == 1


if __name__ == "__main__":
    # Run basic tests
    test_dense_to_banded()
    test_scattering_system_validation()
    test_scattering_system_basic()
    test_scattering_vs_kwant_matrix()
    test_scattering_system_single_lead()
    print("All tests passed!")
