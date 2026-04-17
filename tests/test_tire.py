"""Tests for the Dugoff tire model."""

from __future__ import annotations

import math

import pytest

from abs_sim.physics.tire import DugoffTire, SURFACES, get_surface, slip_risk


def test_zero_slip_gives_zero_force():
    tire = DugoffTire()
    Fx, Fy = tire.forces(kappa=0.0, alpha=0.0, Fz=5000.0, mu=0.9)
    assert Fx == 0.0
    assert Fy == 0.0


def test_zero_load_gives_zero_force():
    tire = DugoffTire()
    Fx, Fy = tire.forces(kappa=0.2, alpha=0.1, Fz=0.0, mu=0.9)
    assert Fx == 0.0
    assert Fy == 0.0


def test_small_slip_is_linear_longitudinal():
    tire = DugoffTire(Cx=80000.0)
    kappa = 0.005
    Fx, _ = tire.forces(kappa=kappa, alpha=0.0, Fz=5000.0, mu=0.9)
    assert Fx == pytest.approx(tire.Cx * kappa, rel=1e-6)


def test_small_slip_angle_is_linear_lateral():
    tire = DugoffTire(Cy=60000.0)
    alpha = 0.01
    _, Fy = tire.forces(kappa=0.0, alpha=alpha, Fz=5000.0, mu=0.9)
    assert Fy == pytest.approx(tire.Cy * math.tan(alpha), rel=1e-6)


def test_braking_gives_negative_fx():
    tire = DugoffTire()
    Fx, _ = tire.forces(kappa=-0.1, alpha=0.0, Fz=5000.0, mu=0.9)
    assert Fx < 0.0


def test_driving_gives_positive_fx():
    tire = DugoffTire()
    Fx, _ = tire.forces(kappa=0.1, alpha=0.0, Fz=5000.0, mu=0.9)
    assert Fx > 0.0


def test_force_magnitude_bounded_by_friction_circle():
    tire = DugoffTire()
    Fz = 5000.0
    mu = 0.9
    # sweep far into saturation in both dimensions
    for kappa in [-0.8, -0.3, 0.1, 0.5, 1.5]:
        for alpha in [-0.4, -0.1, 0.0, 0.2, 0.5]:
            Fx, Fy = tire.forces(kappa, alpha, Fz, mu)
            assert math.hypot(Fx, Fy) <= mu * Fz + 1e-6


def test_longitudinal_curve_has_peak_and_slide():
    # The mu-slip curve should:
    # 1. Saturate towards mu*Fz near slip_peak (~0.15),
    # 2. Fade by slide_ratio at/beyond slip_slide (default 0.5),
    # 3. Never exceed mu*Fz.
    tire = DugoffTire(Cx=80000.0)
    Fz, mu = 5000.0, 0.9
    Fmax = mu * Fz
    samples = {k: abs(tire.pure_longitudinal(k, Fz, mu))
               for k in (0.05, 0.1, 0.15, 0.25, 0.5, 1.0)}
    # Peak near 0.15, well above the 0.05 sample
    assert samples[0.15] > samples[0.05]
    # Fade at full slide to about slide_ratio * Fmax
    assert samples[1.0] == pytest.approx(Fmax * tire.slide_ratio, rel=0.05)
    assert samples[0.5] == pytest.approx(Fmax * tire.slide_ratio, rel=0.05)
    # No sample exceeds Fmax
    for k, v in samples.items():
        assert v <= Fmax + 1e-6, f"Fx at kappa={k} exceeded mu*Fz"
    # Post-peak drop (locked vs peak)
    assert samples[1.0] < samples[0.15]


def test_combined_slip_reduces_lateral_force():
    tire = DugoffTire()
    Fz, mu = 5000.0, 0.9
    _, Fy_pure = tire.forces(kappa=0.0, alpha=0.1, Fz=Fz, mu=mu)
    _, Fy_combined = tire.forces(kappa=-0.2, alpha=0.1, Fz=Fz, mu=mu)
    assert abs(Fy_combined) < abs(Fy_pure)


def test_surface_presets_ordering():
    # Mu should rank: ice < snow < sand < wet < dry.
    order = ["ice", "snow", "sand", "wet", "dry"]
    mus = [SURFACES[s].mu for s in order]
    assert mus == sorted(mus)


def test_get_surface_case_insensitive():
    assert get_surface("DRY").name == "dry"
    with pytest.raises(KeyError):
        get_surface("lava")


def test_slip_risk_bounds_and_monotone():
    assert slip_risk(0.0) == 0.0
    assert slip_risk(0.15) == 0.0
    assert slip_risk(1.0) == 1.0
    assert slip_risk(-1.0) == 1.0
    vals = [slip_risk(k) for k in (0.15, 0.3, 0.5, 0.8, 1.0)]
    assert vals == sorted(vals)
