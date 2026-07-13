"""Unit tests for the inlet turbulence calculator (issue #68).

One seam, key-free and fixture-free (CI runs these with nothing but pytest):
the PURE function (src/turbinlet.py, estimate_turbulence_inlet) — scalars in,
typed estimate out. No filesystem, no temp dirs, no subprocess fakes.

Known-value provenance (the published worked example): the Foundation
OpenFOAM v10 pitzDaily tutorial, shipped verbatim in this repo's tutorial
database (database/raw/openfoam_tutorials_details.txt, first pitzDaily
entry). Its inlet is the standard textbook derivation:

    U = 10 m/s (0/U inlet), I = 5%, l = 10% of the 25.4 mm inlet
    height = 0.00254 m, nu = 1e-5 m^2/s (constant/physicalProperties)

    0/k       freezes k       = 0.375   m^2/s^2   (= 3/2*(10*0.05)^2 exactly)
    0/epsilon freezes epsilon = 14.855  m^2/s^3   (= 0.09^0.75*0.375^1.5/0.00254,
                                                    rounded to 5 significant digits)

Expected values below are those published numbers (or one-line hand
arithmetic FROM them, shown in comments) — never recomputed the module's way.
"""

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import turbinlet  # noqa: E402

# The published pitzDaily inlet (see module docstring for provenance).
PITZ_U = 10.0          # m/s, 0/U inlet value
PITZ_I = 0.05          # 5% intensity
PITZ_L = 0.00254       # m, 10% of the 25.4 mm inlet height
PITZ_NU = 1e-5         # m^2/s, constant/physicalProperties


def _pitz_estimate(**overrides):
    kwargs = dict(velocity=PITZ_U, intensity=PITZ_I, length_scale=PITZ_L)
    kwargs.update(overrides)
    return turbinlet.estimate_turbulence_inlet(**kwargs)


# ---------------------------------------------------------------------------
# Known values: the published pitzDaily inlet quantities
# ---------------------------------------------------------------------------

def test_pitzdaily_k_matches_the_published_tutorial_value():
    # 0/k in the v10 pitzDaily tutorial: uniform 0.375 (exact — the
    # derivation 3/2*(10*0.05)^2 has no rounding).
    estimate = _pitz_estimate()

    assert estimate.k.value == pytest.approx(0.375, rel=1e-9)
    assert estimate.k.units == "m^2/s^2"


def test_pitzdaily_epsilon_matches_the_published_tutorial_value():
    # 0/epsilon in the v10 pitzDaily tutorial: uniform 14.855 (the tutorial
    # rounds to 5 significant digits, hence the 1e-3 tolerance).
    estimate = _pitz_estimate()

    assert estimate.epsilon.value == pytest.approx(14.855, rel=1e-3)
    assert estimate.epsilon.units == "m^2/s^3"


def test_pitzdaily_omega_matches_the_published_pair_via_the_model_identity():
    # Independent route: the k-epsilon/k-omega identity omega = epsilon /
    # (C_mu*k) over the PUBLISHED pair — 14.855/(0.09*0.375) = 440.148...
    # (hand arithmetic, not the module's sqrt(k)/(C_mu^(1/4)*l) path).
    estimate = _pitz_estimate()

    assert estimate.omega.value == pytest.approx(440.148, rel=1e-3)
    assert estimate.omega.units == "1/s"


def test_pitzdaily_nu_t_matches_the_published_pair():
    # nu_t = C_mu*k^2/epsilon over the PUBLISHED pair by hand:
    # 0.09*0.375^2/14.855 = 0.01265625/14.855 = 8.51986e-4 m^2/s.
    estimate = _pitz_estimate()

    assert estimate.nu_t.value == pytest.approx(8.51986e-4, rel=1e-3)
    assert estimate.nu_t.units == "m^2/s"


def test_pitzdaily_viscosity_ratio_with_the_tutorial_nu():
    # nu_t/nu by hand from the published numbers: 8.51986e-4/1e-5 = 85.199 —
    # a healthy figure for a recirculating internal flow (well under the
    # ~O(1000) that flags a pathological combination).
    estimate = _pitz_estimate(kinematic_viscosity=PITZ_NU)

    assert estimate.viscosity_ratio is not None
    assert estimate.viscosity_ratio.value == pytest.approx(85.199, rel=1e-3)
    assert estimate.viscosity_ratio.units == "-"


# ---------------------------------------------------------------------------
# Evidence style: every quantity names the formula that produced it, with the
# pinned constant echoed where it enters (spec #65: the reference cites this
# module as the constants' source)
# ---------------------------------------------------------------------------

def test_c_mu_is_pinned_to_the_standard_value_in_one_place():
    assert turbinlet.C_MU == 0.09
    assert _pitz_estimate().c_mu == 0.09


def test_each_quantity_carries_its_formula_name():
    estimate = _pitz_estimate(kinematic_viscosity=PITZ_NU)

    assert estimate.k.formula == "k = 3/2*(U*I)^2"
    assert estimate.epsilon.formula == \
        "epsilon = C_mu^(3/4)*k^(3/2)/l (C_mu = 0.09)"
    assert estimate.omega.formula == \
        "omega = sqrt(k)/(C_mu^(1/4)*l) (C_mu = 0.09)"
    assert estimate.nu_t.formula == "nu_t = C_mu*k^2/epsilon (C_mu = 0.09)"
    assert estimate.viscosity_ratio.formula == "viscosity_ratio = nu_t/nu"


# ---------------------------------------------------------------------------
# Stated assumptions: an omitted intensity applies the documented medium
# default and ECHOES it; a hydraulic diameter converts via the named rule
# ---------------------------------------------------------------------------

def test_omitted_intensity_applies_and_echoes_the_medium_default():
    # The documented medium default is 0.05 — exactly the pitzDaily I, so the
    # published k value must reappear unchanged.
    estimate = turbinlet.estimate_turbulence_inlet(
        velocity=PITZ_U, length_scale=PITZ_L)

    assert turbinlet.DEFAULT_INTENSITY == 0.05
    assert estimate.intensity == 0.05
    assert estimate.k.value == pytest.approx(0.375, rel=1e-9)
    assert "default" in estimate.intensity_source
    assert "0.05" in estimate.intensity_source
    joined = "\n".join(estimate.assumptions)
    assert "0.05" in joined and "default" in joined


def test_caller_supplied_intensity_is_named_as_such_with_no_assumption():
    estimate = _pitz_estimate()

    assert estimate.intensity_source == "caller-supplied"
    assert estimate.length_scale_source == "caller-supplied"
    assert estimate.assumptions == []


def test_hydraulic_diameter_converts_via_the_named_standard_rule():
    # l = 0.07*D_h (hand: 0.07*0.1 = 0.007 m) — and the SAME numbers as an
    # explicit length_scale=0.007 call, so the conversion is the only
    # difference between the two spellings.
    via_diameter = turbinlet.estimate_turbulence_inlet(
        velocity=PITZ_U, intensity=PITZ_I, hydraulic_diameter=0.1)
    via_length = turbinlet.estimate_turbulence_inlet(
        velocity=PITZ_U, intensity=PITZ_I, length_scale=0.007)

    assert via_diameter.length_scale == pytest.approx(0.007, rel=1e-12)
    assert via_diameter.k.value == pytest.approx(via_length.k.value, rel=1e-12)
    assert via_diameter.epsilon.value == \
        pytest.approx(via_length.epsilon.value, rel=1e-12)
    assert via_diameter.omega.value == \
        pytest.approx(via_length.omega.value, rel=1e-12)
    assert "0.07*D_h" in via_diameter.length_scale_source
    assert "0.07*D_h" in "\n".join(via_diameter.assumptions)


def test_omitted_viscosity_omits_the_ratio_instead_of_assuming_a_fluid():
    # The fluid is the caller's fact: without nu there is no ratio, and the
    # other quantities are unaffected.
    estimate = _pitz_estimate()

    assert estimate.viscosity_ratio is None
    assert estimate.nu_t.value == pytest.approx(8.51986e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# Properties that pin trust in the arithmetic
# ---------------------------------------------------------------------------

def test_k_scales_with_the_square_of_velocity_times_intensity():
    # k = 3/2*(U*I)^2: doubling U (or I) must exactly quadruple k.
    base = turbinlet.estimate_turbulence_inlet(
        velocity=2.0, intensity=0.04, length_scale=0.5)
    double_u = turbinlet.estimate_turbulence_inlet(
        velocity=4.0, intensity=0.04, length_scale=0.5)
    double_i = turbinlet.estimate_turbulence_inlet(
        velocity=2.0, intensity=0.08, length_scale=0.5)

    assert double_u.k.value == pytest.approx(4 * base.k.value, rel=1e-12)
    assert double_i.k.value == pytest.approx(4 * base.k.value, rel=1e-12)


def test_epsilon_and_omega_scale_inversely_with_the_length_scale():
    # Both carry l in the denominator: doubling l halves each, k unchanged.
    base = turbinlet.estimate_turbulence_inlet(
        velocity=5.0, intensity=0.05, length_scale=0.1)
    double_l = turbinlet.estimate_turbulence_inlet(
        velocity=5.0, intensity=0.05, length_scale=0.2)

    assert double_l.k.value == pytest.approx(base.k.value, rel=1e-12)
    assert double_l.epsilon.value == \
        pytest.approx(base.epsilon.value / 2, rel=1e-12)
    assert double_l.omega.value == \
        pytest.approx(base.omega.value / 2, rel=1e-12)


# ---------------------------------------------------------------------------
# Typed errors: ambiguity and non-physical inputs fail loudly, never
# plausible garbage numbers
# ---------------------------------------------------------------------------

def test_neither_length_scale_nor_hydraulic_diameter_is_a_typed_error():
    with pytest.raises(turbinlet.TurbulenceInletError, match="exactly one"):
        turbinlet.estimate_turbulence_inlet(velocity=PITZ_U, intensity=PITZ_I)


def test_both_length_scale_and_hydraulic_diameter_is_a_typed_error():
    with pytest.raises(turbinlet.TurbulenceInletError, match="exactly one"):
        turbinlet.estimate_turbulence_inlet(
            velocity=PITZ_U, intensity=PITZ_I,
            length_scale=PITZ_L, hydraulic_diameter=0.1)


@pytest.mark.parametrize("bad_velocity", [0.0, -3.0, float("nan"), float("inf")])
def test_non_physical_velocity_is_a_typed_error(bad_velocity):
    with pytest.raises(turbinlet.TurbulenceInletError, match="velocity"):
        turbinlet.estimate_turbulence_inlet(
            velocity=bad_velocity, intensity=PITZ_I, length_scale=PITZ_L)


@pytest.mark.parametrize("name,kwargs", [
    ("intensity", dict(intensity=0.0, length_scale=PITZ_L)),
    ("intensity", dict(intensity=-0.05, length_scale=PITZ_L)),
    ("length_scale", dict(intensity=PITZ_I, length_scale=0.0)),
    ("length_scale", dict(intensity=PITZ_I, length_scale=-1.0)),
    ("hydraulic_diameter", dict(intensity=PITZ_I, hydraulic_diameter=0.0)),
    ("hydraulic_diameter", dict(intensity=PITZ_I, hydraulic_diameter=-0.1)),
    ("kinematic_viscosity",
     dict(intensity=PITZ_I, length_scale=PITZ_L, kinematic_viscosity=0.0)),
    ("kinematic_viscosity",
     dict(intensity=PITZ_I, length_scale=PITZ_L, kinematic_viscosity=-1e-5)),
])
def test_non_physical_optional_inputs_are_typed_errors_naming_the_input(
        name, kwargs):
    with pytest.raises(turbinlet.TurbulenceInletError, match=name):
        turbinlet.estimate_turbulence_inlet(velocity=PITZ_U, **kwargs)
