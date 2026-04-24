"""
tests/test_section.py
=====================
Tests for SectionState, Section (ABC), ElasticSection.
"""

import pytest
import numpy as np
from material import ElasticMaterial, MaterialState
from section import SectionState, ElasticSection


# ---------------------------------------------------------------------------
# SectionState
# ---------------------------------------------------------------------------

class TestSectionState:

    def test_initial_values_are_zero(self):
        s = SectionState()
        assert s.eps_committed == 0.0
        assert s.eps_trial == 0.0
        assert s.kappa_committed == 0.0
        assert s.kappa_trial == 0.0
        assert s.fiber_states == []

    def test_commit_copies_trial_to_committed(self):
        s = SectionState()
        s.eps_trial = 0.001
        s.kappa_trial = 0.002
        s.commit()
        assert s.eps_committed == 0.001
        assert s.kappa_committed == 0.002

    def test_revert_restores_committed(self):
        s = SectionState()
        s.eps_trial = 0.001
        s.kappa_trial = 0.002
        s.commit()
        s.eps_trial = 0.010
        s.kappa_trial = 0.020
        s.revert()
        assert s.eps_trial == 0.001
        assert s.kappa_trial == 0.002

    def test_commit_delegates_to_fiber_states(self):
        s = SectionState()
        fs = MaterialState()
        fs.eps_trial = 0.005
        s.fiber_states.append(fs)
        s.commit()
        assert fs.eps_committed == 0.005

    def test_revert_delegates_to_fiber_states(self):
        s = SectionState()
        fs = MaterialState()
        fs.eps_trial = 0.005
        fs.commit()
        fs.eps_trial = 0.999
        s.fiber_states.append(fs)
        s.revert()
        assert fs.eps_trial == 0.005

    def test_serialization_round_trip(self):
        s = SectionState()
        s.eps_trial = 0.001
        s.kappa_trial = 0.002
        s.commit()
        s2 = SectionState.from_dict(s.to_dict())
        assert s2.eps_committed == s.eps_committed
        assert s2.kappa_committed == s.kappa_committed

    def test_serialization_with_fiber_states(self):
        s = SectionState()
        fs = MaterialState(eps_trial=0.003, sig_trial=630.0)
        s.fiber_states.append(fs)
        s2 = SectionState.from_dict(s.to_dict())
        assert len(s2.fiber_states) == 1
        assert s2.fiber_states[0].eps_trial == 0.003


# ---------------------------------------------------------------------------
# ElasticSection
# ---------------------------------------------------------------------------

class TestElasticSection:

    def setup_method(self):
        self.mat = ElasticMaterial(id="steel", E=210_000, rho=7.85e-3)
        self.sec = ElasticSection(
            id="IPE300", material_id="steel", A=53.8, I=8360.0
        )

    def test_get_stiffness_EA_EI(self):
        EA, EI = self.sec.get_stiffness(self.mat)
        assert EA == pytest.approx(210_000.0 * 53.8)
        assert EI == pytest.approx(210_000.0 * 8360.0)

    def test_negative_A_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            ElasticSection(id="bad", material_id="m", A=-1.0, I=100.0)

    def test_zero_A_raises_value_error(self):
        with pytest.raises(ValueError):
            ElasticSection(id="bad", material_id="m", A=0.0, I=100.0)

    def test_negative_I_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            ElasticSection(id="bad", material_id="m", A=10.0, I=-1.0)

    def test_get_local_mass_shape(self):
        M = self.sec.get_local_mass(self.mat, L=5.0)
        assert M.shape == (6, 6)

    def test_get_local_mass_is_symmetric(self):
        M = self.sec.get_local_mass(self.mat, L=5.0)
        np.testing.assert_allclose(M, M.T, atol=1e-12)

    def test_get_local_mass_is_positive_definite(self):
        M = self.sec.get_local_mass(self.mat, L=5.0)
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues > 0)

    def test_get_local_mass_total_mass(self):
        # total mass = rho * A * L
        L = 5.0
        M = self.sec.get_local_mass(self.mat, L=L)
        expected_mass = self.mat.rho * self.sec.A * L
        # sum of diagonal translational DOFs (u_i, v_i, u_j, v_j)
        # for a consistent matrix equals the total element mass
        translational = [0, 1, 3, 4]
        assert M[0, 0] + M[3, 3] == pytest.approx(
            expected_mass * (140 + 140) / 420 * 2 / 2, rel=1e-6
        )

    def test_get_local_mass_zero_rho_gives_zero_matrix(self):
        mat_no_rho = ElasticMaterial(id="no_rho", E=210_000.0, rho=0.0)
        M = self.sec.get_local_mass(mat_no_rho, L=5.0)
        np.testing.assert_allclose(M, np.zeros((6, 6)))

    def test_serialization_round_trip(self):
        d = self.sec.to_dict()
        assert d["type"] == "ElasticSection"
        sec2 = ElasticSection.from_dict(d)
        assert sec2.id == self.sec.id
        assert sec2.material_id == self.sec.material_id
        assert sec2.A == self.sec.A
        assert sec2.I == self.sec.I

    def test_optional_unit(self):
        sec = ElasticSection(
            id="s", material_id="m", A=10.0, I=100.0, unit="cm"
        )
        sec2 = ElasticSection.from_dict(sec.to_dict())
        assert sec2.unit == "cm"

    def test_repr(self):
        r = repr(self.sec)
        assert "ElasticSection" in r
        assert "IPE300" in r
