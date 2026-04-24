"""
tests/test_material.py
======================
Tests for MaterialState, Material (ABC), ElasticMaterial.
"""

import pytest
from material import MaterialState, ElasticMaterial


# ---------------------------------------------------------------------------
# MaterialState
# ---------------------------------------------------------------------------

class TestMaterialState:

    def test_initial_values_are_zero(self):
        s = MaterialState()
        assert s.eps_committed == 0.0
        assert s.sig_committed == 0.0
        assert s.eps_trial == 0.0
        assert s.sig_trial == 0.0

    def test_commit_copies_trial_to_committed(self):
        s = MaterialState()
        s.eps_trial = 0.001
        s.sig_trial = 210.0
        s.eps_p_trial = 0.0005
        s.alpha_trial = 1.5
        s.commit()
        assert s.eps_committed == 0.001
        assert s.sig_committed == 210.0
        assert s.eps_p_committed == 0.0005
        assert s.alpha_committed == 1.5

    def test_revert_restores_committed(self):
        s = MaterialState()
        s.eps_trial = 0.001
        s.sig_trial = 210.0
        s.commit()
        s.eps_trial = 0.005
        s.sig_trial = 1050.0
        s.revert()
        assert s.eps_trial == 0.001
        assert s.sig_trial == 210.0

    def test_commit_then_revert_preserves_data(self):
        s = MaterialState()
        s.eps_trial = 0.002
        s.commit()
        s.eps_trial = 0.010
        s.revert()
        s.commit()
        assert s.eps_committed == 0.002

    def test_serialization_round_trip(self):
        s = MaterialState(
            eps_committed=0.001, sig_committed=210.0,
            eps_trial=0.002,     sig_trial=420.0,
        )
        s2 = MaterialState.from_dict(s.to_dict())
        assert s2.eps_committed == s.eps_committed
        assert s2.sig_committed == s.sig_committed
        assert s2.eps_trial == s.eps_trial
        assert s2.sig_trial == s.sig_trial


# ---------------------------------------------------------------------------
# ElasticMaterial
# ---------------------------------------------------------------------------

class TestElasticMaterial:

    def setup_method(self):
        self.mat = ElasticMaterial(id="steel", E=210_000.0, rho=7.85e-3)
        self.state = MaterialState()

    def test_compute_hookes_law(self):
        sigma, E_tan = self.mat.compute(eps=0.001, state=self.state)
        assert sigma == pytest.approx(210.0)
        assert E_tan == 210_000.0

    def test_compute_negative_strain(self):
        sigma, E_tan = self.mat.compute(eps=-0.001, state=self.state)
        assert sigma == pytest.approx(-210.0)
        assert E_tan == 210_000.0

    def test_compute_zero_strain(self):
        sigma, E_tan = self.mat.compute(eps=0.0, state=self.state)
        assert sigma == 0.0
        assert E_tan == 210_000.0

    def test_compute_does_not_modify_state(self):
        self.state.eps_committed = 0.001
        self.mat.compute(eps=0.005, state=self.state)
        assert self.state.eps_committed == 0.001
        assert self.state.eps_trial == 0.0

    def test_negative_E_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            ElasticMaterial(id="bad", E=-1.0)

    def test_zero_E_raises_value_error(self):
        with pytest.raises(ValueError):
            ElasticMaterial(id="bad", E=0.0)

    def test_serialization_round_trip(self):
        d = self.mat.to_dict()
        assert d["type"] == "ElasticMaterial"
        mat2 = ElasticMaterial.from_dict(d)
        assert mat2.id == self.mat.id
        assert mat2.E == self.mat.E
        assert mat2.rho == self.mat.rho

    def test_repr(self):
        r = repr(self.mat)
        assert "ElasticMaterial" in r
        assert "steel" in r

    def test_optional_unit(self):
        mat = ElasticMaterial(id="concrete", E=30_000, unit="kN/cm²")
        assert mat.unit == "kN/cm²"
        mat2 = ElasticMaterial.from_dict(mat.to_dict())
        assert mat2.unit == "kN/cm²"

    def test_default_rho_is_zero(self):
        mat = ElasticMaterial(id="concrete", E=30_000)
        assert mat.rho == 0.0
