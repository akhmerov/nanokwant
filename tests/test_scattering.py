"""Tests for the scattering module."""

import numpy as np
import pytest
import kwant
from nanokwant.scattering import scattering_system, compute_smatrix


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

    # System with negative hopping (not in Hermitian format)
    system_negative = {
        0: {"mu": np.eye(2)},
        1: {"t": np.eye(2)},
        -1: {"t": np.eye(2)},
    }

    with pytest.raises(ValueError, match="Hermitian format"):
        scattering_system(system_negative, 10, {"mu": 1.0, "t": 1.0}, energy=0.0)

    # System without hopping
    system_no_hop = {
        0: {"mu": np.eye(2)},
    }

    with pytest.raises(ValueError, match="nearest-neighbor hopping"):
        scattering_system(system_no_hop, 10, {"mu": 1.0}, energy=0.0)


def test_scattering_without_onsite():
    """Test that scattering works without onsite terms (zero onsite is OK)."""
    # System with only hopping, no onsite
    system = {
        1: {"t": np.eye(2)},
    }
    num_sites = 5
    params = {"t": 1.0}
    energy = 0.5

    # This should work now
    lhs, (l, u), rhs_list, indices_list, nmodes_list = scattering_system(
        system, num_sites, params, energy, leads="both"
    )

    assert lhs.ndim == 2
    assert l >= 0 and u >= 0
    assert len(rhs_list) == 2
    assert len(indices_list) == 2
    assert len(nmodes_list) == 2


def test_scattering_system_basic():
    """Test basic construction of a scattering system."""
    # Simple 1D system with constant parameters
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": np.eye(2)},
    }
    num_sites = 5
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5

    lhs, (l, u), rhs_list, indices_list, nmodes_list = scattering_system(
        system, num_sites, params, energy, leads="both"
    )

    # Check shapes
    assert lhs.ndim == 2
    assert l >= 0 and u >= 0

    # Should have 2 leads (left and right)
    assert len(rhs_list) == 2
    assert len(indices_list) == 2
    assert len(nmodes_list) == 2

    # Each RHS should be a matrix
    for rhs, nmodes in zip(rhs_list, nmodes_list):
        assert rhs.ndim == 2
        assert rhs.shape[0] == lhs.shape[1]
        assert rhs.shape[1] == nmodes


def test_scattering_system_single_lead():
    """Test scattering system with only one lead."""
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": np.eye(2)},
    }
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5

    # Left lead only
    lhs_left, (l, u), rhs_left, idx_left, nmodes_left = scattering_system(
        system, 5, params, energy, leads="left"
    )
    assert len(rhs_left) == 1
    assert len(idx_left) == 1
    assert len(nmodes_left) == 1

    # Right lead only
    lhs_right, (l, u), rhs_right, idx_right, nmodes_right = scattering_system(
        system, 5, params, energy, leads="right"
    )
    assert len(rhs_right) == 1
    assert len(idx_right) == 1
    assert len(nmodes_right) == 1


def test_smatrix_computation_simple():
    """Test S-matrix computation with simple system."""
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": np.eye(2)},
    }
    num_sites = 10
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.5

    # Compute S-matrix
    S = compute_smatrix(system, num_sites, params, energy, leads="both")

    # Should be square and unitary
    assert S.shape[0] == S.shape[1]
    np.testing.assert_allclose(S @ S.conj().T, np.eye(S.shape[0]), atol=1e-10)

    # Compare with Kwant
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()

    for i in range(num_sites):
        syst[lat(i)] = params["mu"] * np.eye(2)
    syst[lat.neighbors()] = params["t"] * np.eye(2)

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = params["mu"] * np.eye(2)
    lead[lat.neighbors()] = params["t"] * np.eye(2)

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    fsyst = syst.finalized()
    smat_kwant = kwant.smatrix(fsyst, energy)

    # Check that shapes match
    assert S.shape == smat_kwant.data.shape

    # Check that S-matrix magnitude matches
    np.testing.assert_allclose(
        np.abs(S), np.abs(smat_kwant.data), rtol=1e-5, atol=1e-10
    )


def test_smatrix_with_evanescent_modes():
    """Test S-matrix with evanescent modes (different bandwidth orbitals)."""
    # Use different hopping for different orbitals to create evanescent modes
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": np.diag([1.0, 0.1])},  # First orbital: high bandwidth, second: low
    }
    num_sites = 10
    params = {"mu": 0.5, "t": 1.0}
    energy = 0.3  # Energy where second orbital has evanescent modes

    # This should work and handle evanescent modes correctly
    S = compute_smatrix(system, num_sites, params, energy, leads="both")

    # S-matrix should be unitary
    np.testing.assert_allclose(S @ S.conj().T, np.eye(S.shape[0]), atol=1e-10)

    # Compare with Kwant
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()

    for i in range(num_sites):
        syst[lat(i)] = params["mu"] * np.eye(2)
    syst[lat.neighbors()] = params["t"] * np.diag([1.0, 0.1])

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = params["mu"] * np.eye(2)
    lead[lat.neighbors()] = params["t"] * np.diag([1.0, 0.1])

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    fsyst = syst.finalized()
    smat_kwant = kwant.smatrix(fsyst, energy)

    assert S.shape == smat_kwant.data.shape
    np.testing.assert_allclose(
        np.abs(S), np.abs(smat_kwant.data), rtol=1e-5, atol=1e-10
    )


@pytest.mark.parametrize("norbs", [2, 3])
@pytest.mark.parametrize("complex_hopping", [False, True])
def test_smatrix_randomized(norbs, complex_hopping):
    """Randomized test with various matrix sizes and complex hoppings."""
    rng = np.random.default_rng(42)

    # Generate random Hermitian onsite
    onsite = rng.standard_normal((norbs, norbs))
    onsite = onsite + onsite.T  # Make symmetric (will become Hermitian)

    # Generate random hopping
    if complex_hopping:
        hopping = rng.standard_normal((norbs, norbs)) + 1j * rng.standard_normal(
            (norbs, norbs)
        )
    else:
        hopping = rng.standard_normal((norbs, norbs))

    system = {
        0: {"mu": onsite},
        1: {"t": hopping},
    }
    num_sites = 8
    params = {"mu": 1.0, "t": 1.0}
    energy = 0.5

    # Compute S-matrix
    S = compute_smatrix(system, num_sites, params, energy, leads="both")

    # Should be unitary
    np.testing.assert_allclose(S @ S.conj().T, np.eye(S.shape[0]), atol=1e-9)

    # Compare with Kwant
    lat = kwant.lattice.chain(norbs=norbs)
    syst = kwant.Builder()

    for i in range(num_sites):
        syst[lat(i)] = params["mu"] * onsite
    syst[lat.neighbors()] = params["t"] * hopping

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = params["mu"] * onsite
    lead[lat.neighbors()] = params["t"] * hopping

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    fsyst = syst.finalized()
    smat_kwant = kwant.smatrix(fsyst, energy)

    assert S.shape == smat_kwant.data.shape
    np.testing.assert_allclose(np.abs(S), np.abs(smat_kwant.data), rtol=1e-4, atol=1e-8)


def test_smatrix_position_dependent():
    """Test S-matrix with position-dependent parameters."""
    # Use position-dependent onsite potential
    system = {
        0: {"mu": np.eye(2)},
        1: {"t": np.eye(2)},
    }
    num_sites = 10

    # Position-dependent chemical potential - barrier in the middle
    def mu_func(x):
        return np.where((x >= 4) & (x <= 6), 2.0, 0.5)

    params = {"mu": mu_func, "t": 1.0}
    energy = 0.5

    # Compute S-matrix
    S = compute_smatrix(system, num_sites, params, energy, leads="both")

    # Should be unitary
    np.testing.assert_allclose(S @ S.conj().T, np.eye(S.shape[0]), atol=1e-10)

    # Compare with Kwant
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()

    for i in range(num_sites):
        mu_val = mu_func(np.array([i]))[0]
        syst[lat(i)] = mu_val * np.eye(2)
    syst[lat.neighbors()] = params["t"] * np.eye(2)

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = params["mu"](np.array([0]))[0] * np.eye(2)  # Use edge value
    lead[lat.neighbors()] = params["t"] * np.eye(2)

    syst.attach_lead(lead)

    # For right lead, use the edge value
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    lead_right[lat(0)] = params["mu"](np.array([num_sites - 1]))[0] * np.eye(2)
    lead_right[lat.neighbors()] = params["t"] * np.eye(2)
    syst.attach_lead(lead_right.reversed())

    fsyst = syst.finalized()
    smat_kwant = kwant.smatrix(fsyst, energy)

    assert S.shape == smat_kwant.data.shape
    np.testing.assert_allclose(
        np.abs(S), np.abs(smat_kwant.data), rtol=1e-5, atol=1e-10
    )


if __name__ == "__main__":
    test_scattering_system_validation()
    test_scattering_without_onsite()
    test_scattering_system_basic()
    test_scattering_system_single_lead()
    test_smatrix_computation_simple()
    test_smatrix_with_evanescent_modes()
    test_smatrix_randomized(2, False)
    test_smatrix_randomized(3, True)
    test_smatrix_position_dependent()
    print("All tests passed!")
