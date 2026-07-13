"""Unit tests for the wall-spacing calculator (issue #67).

One seam, key-free and dependency-free (CI runs these with nothing but
pytest): the PURE calculator ``estimate_wall_spacing`` in src/wallspacing.py,
judged only on its typed result for given scalar inputs. The first purely
computational tool in the server: no fixtures, no temp dirs, no subprocess
fakes anywhere in this module.

Known-value expectations are frozen literals hand-derived from the PUBLISHED
correlations, never recomputed the module's way. Correlation provenance
(each verified 2026-07-13 against the named published sources):

- External turbulent Cf: Schlichting flat-plate local skin friction,
  Cf = (2*log10(Re_x) - 0.65)^(-2.3), valid Re_x < 1e9 (Schlichting,
  Boundary-Layer Theory; printed identically on the Quadco Engineering and
  Flowthermolab y+ calculator pages).
- External laminar: Blasius flat-plate similarity solution (exact),
  Cf = 0.664*Re_x^(-1/2), delta = 5.0*x/sqrt(Re_x) (White, Fluid Mechanics;
  Incropera, Fundamentals of Heat and Mass Transfer, ch. 7).
- Internal turbulent: Blasius smooth-pipe friction factor,
  f = 0.316*Re_D^(-1/4) (Darcy), valid 4000 < Re_D < 1e5, tau_w = (f/8)*rho*U^2
  (Flowthermolab page; the LEAP CFD page prints the same correlation in
  Fanning form, Cf = 0.079*Re^(-0.25) = f/4 — two sources agreeing).
  Independent anchor: the Moody chart's smooth-pipe value f ~ 0.018 at
  Re = 1e5 (White, Fluid Mechanics).
- Internal laminar: Hagen-Poiseuille fully developed pipe flow (exact),
  f = 64/Re_D; independently tau_w/rho = 8*nu*U/D from the parabolic
  profile — two exact derivation paths that must agree.
- The spacing chain (all four sources print it identically):
  tau_w = (Cf/2)*rho*U^2, u_tau = sqrt(tau_w/rho), y1 = y+*nu/u_tau.
- Independent anchor for the external chain: the one-fifth-power-law local
  skin friction Cf = 0.0592*Re_x^(-1/5), valid 5e5 < Re_x < 1e7 (Incropera,
  Fundamentals of Heat and Mass Transfer, eq. 7.34) — a DIFFERENT published
  correlation from the pinned Schlichting log law, so a shared misreading
  cannot make both agree.

Regime thresholds under test: external flat plate laminar below Re_x = 5e5
(transition onset; Incropera sec. 7.2), fully turbulent at/above Re_x = 3e6
(White, Fluid Mechanics: transition band 5e5..3e6); internal pipe laminar
below Re_D = 2300, turbulent at/above Re_D = 4000, transitional between
(Incropera sec. 8.1; Cengel & Cimbala).

MCP wrapper + registration checks live in test_mcp_helpers.py (importorskip
pattern — fastmcp is not installed in CI).
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import wallspacing  # noqa: E402


# ---------------------------------------------------------------------------
# Known value: external turbulent flat plate (Schlichting Cf)
# Air-like case: U = 50 m/s, L = 1 m, nu = 1.5e-5 m^2/s, y+ = 1, ratio 1.2.
# Hand-derived 2026-07-13 from the published formulas (module docstring):
#   Re_x      = 50*1/1.5e-5                      = 3.33333e6   (turbulent)
#   Cf        = (2*log10(Re_x) - 0.65)^(-2.3)    = 3.05824e-3
#   tau_w/rho = (Cf/2)*U^2                       = 3.8228 m^2/s^2
#   u_tau     = sqrt(tau_w/rho)                  = 1.9552 m/s
#   y1        = y+*nu/u_tau                      = 7.67185e-6 m (cell CENTRE)
#   h1        = 2*y1                             = 1.53437e-5 m (cell HEIGHT)
#   delta     = 0.37*L*Re_x^(-1/5)               = 1.83496e-2 m
#   N         = ceil(ln(1 + delta*(r-1)/h1)/ln r) = 31 layers at r = 1.2
# ---------------------------------------------------------------------------

def _external_turbulent():
    return wallspacing.estimate_wall_spacing(
        velocity=50.0, characteristic_length=1.0,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
        flow_type="external", expansion_ratio=1.2,
    )


def test_external_turbulent_reynolds_and_regime():
    estimate = _external_turbulent()

    assert estimate.reynolds_number.value == pytest.approx(3.33333e6, rel=1e-4)
    assert estimate.regime == "turbulent"
    assert estimate.flow_type == "external"


def test_external_turbulent_skin_friction_via_schlichting():
    estimate = _external_turbulent()

    assert estimate.skin_friction_coefficient.value == pytest.approx(
        3.05824e-3, rel=1e-4)
    assert "Schlichting" in estimate.skin_friction_coefficient.formula


def test_external_turbulent_shear_and_friction_velocity():
    estimate = _external_turbulent()

    assert estimate.kinematic_wall_shear_stress.value == pytest.approx(
        3.8228, rel=1e-4)
    assert estimate.friction_velocity.value == pytest.approx(1.9552, rel=1e-4)


def test_external_turbulent_first_cell_centre_and_height():
    estimate = _external_turbulent()

    assert estimate.first_cell_centre_distance.value == pytest.approx(
        7.67185e-6, rel=1e-4)
    assert estimate.first_cell_height.value == pytest.approx(
        1.53437e-5, rel=1e-4)


def test_external_turbulent_boundary_layer_and_layer_count():
    estimate = _external_turbulent()

    assert estimate.boundary_layer_thickness.value == pytest.approx(
        1.83496e-2, rel=1e-4)
    assert estimate.suggested_layer_count.value == 31


def test_external_turbulent_matches_the_power_law_anchor():
    # Independent published anchor (module docstring): the one-fifth-power
    # law Cf = 0.0592*Re_x^(-1/5) — Incropera eq. 7.34, a different
    # published correlation than the pinned Schlichting log law — gives
    # 0.0592*(3.33333e6)^(-0.2) = 2.93593e-3 at the known-value Re_x.
    # Inside their common validity window the two correlations agree to a
    # few percent (4.2% here); a shared misreading of either could not
    # keep them this close.
    estimate = _external_turbulent()

    assert estimate.skin_friction_coefficient.value == pytest.approx(
        2.93593e-3, rel=0.05)


def test_every_numeric_field_names_its_formula():
    # Evidence style (assess_mesh / inspect_stl precedent): a number without
    # the name of the correlation/formula that produced it is not returned.
    estimate = _external_turbulent()

    for quantity in (
        estimate.reynolds_number,
        estimate.skin_friction_coefficient,
        estimate.kinematic_wall_shear_stress,
        estimate.friction_velocity,
        estimate.first_cell_centre_distance,
        estimate.first_cell_height,
        estimate.boundary_layer_thickness,
        estimate.suggested_layer_count,
    ):
        assert quantity.formula.strip()


def test_centre_and_height_are_separately_labelled():
    # The factor-of-2 confusion dies in the schema: the two fields name
    # themselves unambiguously in their formula strings.
    estimate = _external_turbulent()

    assert "CENTRE" in estimate.first_cell_centre_distance.formula.upper()
    assert "HEIGHT" in estimate.first_cell_height.formula.upper()


# ---------------------------------------------------------------------------
# Known value: internal turbulent pipe (Blasius pipe f)
# Water-like case: U = 2 m/s, D_h = 0.05 m, nu = 1e-6 m^2/s, y+ = 30, r = 1.2.
# Hand-derived 2026-07-13 from the published formulas (module docstring):
#   Re_D      = 2*0.05/1e-6              = 1.0e5      (turbulent)
#   f (Darcy) = 0.316*Re_D^(-1/4)        = 1.77699e-2
#   Cf = f/4  (Fanning)                  = 4.4425e-3
#   tau_w/rho = (f/8)*U^2 = (Cf/2)*U^2   = 8.88499e-3 m^2/s^2
#   u_tau     = sqrt(tau_w/rho)          = 9.42602e-2 m/s
#   y1        = 30*1e-6/u_tau            = 3.18268e-4 m (cell CENTRE)
#   h1        = 2*y1                     = 6.36536e-4 m (cell HEIGHT)
#   delta     = D_h/2                    = 0.025 m
#   N         = 12 layers at r = 1.2
# Independent anchors: the Moody chart's smooth-pipe f ~ 0.018 at Re = 1e5
# (White, Fluid Mechanics), and the LEAP CFD page's Fanning form
# 0.079*Re^(-0.25), which must equal f/4 exactly.
# ---------------------------------------------------------------------------

def _internal_turbulent():
    return wallspacing.estimate_wall_spacing(
        velocity=2.0, characteristic_length=0.05,
        kinematic_viscosity=1.0e-6, target_y_plus=30.0,
        flow_type="internal", expansion_ratio=1.2,
    )


def test_internal_turbulent_reynolds_and_regime():
    estimate = _internal_turbulent()

    assert estimate.reynolds_number.value == pytest.approx(1.0e5, rel=1e-6)
    assert estimate.regime == "turbulent"


def test_internal_turbulent_skin_friction_via_blasius_pipe():
    estimate = _internal_turbulent()

    assert estimate.skin_friction_coefficient.value == pytest.approx(
        4.4425e-3, rel=1e-4)
    assert "Blasius" in estimate.skin_friction_coefficient.formula
    # Fanning vs Darcy is the other classic silent factor bug — the
    # convention must be named where the number travels.
    assert "Fanning" in estimate.skin_friction_coefficient.formula


def test_internal_turbulent_matches_the_moody_chart_anchor():
    # Independent published anchor: the Moody chart's smooth-pipe friction
    # factor at Re = 1e5 is ~0.018 (White, Fluid Mechanics). The returned
    # Fanning Cf times 4 must land on it within 3%.
    estimate = _internal_turbulent()

    assert 4.0 * estimate.skin_friction_coefficient.value == pytest.approx(
        0.018, rel=0.03)


def test_internal_turbulent_spacing_chain():
    estimate = _internal_turbulent()

    assert estimate.kinematic_wall_shear_stress.value == pytest.approx(
        8.88499e-3, rel=1e-4)
    assert estimate.friction_velocity.value == pytest.approx(
        9.42602e-2, rel=1e-4)
    assert estimate.first_cell_centre_distance.value == pytest.approx(
        3.18268e-4, rel=1e-4)
    assert estimate.first_cell_height.value == pytest.approx(
        6.36536e-4, rel=1e-4)


def test_internal_boundary_layer_is_the_radius_with_layer_count():
    estimate = _internal_turbulent()

    assert estimate.boundary_layer_thickness.value == pytest.approx(0.025)
    assert estimate.suggested_layer_count.value == 12


# ---------------------------------------------------------------------------
# Known value: external laminar flat plate (Blasius, exact)
# U = 0.5 m/s, L = 0.5 m, nu = 1.5e-5 m^2/s, y+ = 1, r = 1.2.
# Hand-derived 2026-07-13 from the exact Blasius solution:
#   Re_x = 0.5*0.5/1.5e-5      = 1.66667e4  (laminar, < 5e5)
#   Cf   = 0.664/sqrt(Re_x)    = 5.14332e-3
#   u_tau = U*sqrt(Cf/2)       = 2.53558e-2 m/s
#   y1   = 1*1.5e-5/u_tau      = 5.91581e-4 m
#   delta = 5.0*L/sqrt(Re_x)   = 1.93649e-2 m
# ---------------------------------------------------------------------------

def _external_laminar():
    return wallspacing.estimate_wall_spacing(
        velocity=0.5, characteristic_length=0.5,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
        flow_type="external", expansion_ratio=1.2,
    )


def test_external_laminar_regime_still_returns_all_numbers():
    # A laminar case is not an error: every number comes back (from the
    # laminar correlation) and the verdict carries the inapplicability.
    estimate = _external_laminar()

    assert estimate.regime == "laminar"
    assert estimate.reynolds_number.value == pytest.approx(1.66667e4, rel=1e-4)
    assert estimate.skin_friction_coefficient.value == pytest.approx(
        5.14332e-3, rel=1e-4)
    assert "Blasius" in estimate.skin_friction_coefficient.formula
    assert estimate.friction_velocity.value == pytest.approx(
        2.53558e-2, rel=1e-4)
    assert estimate.first_cell_centre_distance.value == pytest.approx(
        5.91581e-4, rel=1e-4)
    assert estimate.boundary_layer_thickness.value == pytest.approx(
        1.93649e-2, rel=1e-4)


def test_laminar_verdict_names_the_inapplicability():
    estimate = _external_laminar()

    evidence = "\n".join(estimate.evidence)
    assert "laminar" in evidence
    assert "wall functions" in evidence
    assert "inapplicable" in evidence


# ---------------------------------------------------------------------------
# Known value: internal laminar pipe (Hagen-Poiseuille, exact, two paths)
# U = 0.02 m/s, D_h = 0.1 m, nu = 1e-6 m^2/s -> Re_D = 2000 (laminar).
#   f = 64/Re_D = 0.032; Cf = f/4 = 0.008
#   tau_w/rho = (Cf/2)*U^2 = 1.6e-6 m^2/s^2
# Independent exact cross-check straight from the parabolic profile:
#   tau_w/rho = 8*nu*U/D = 8*1e-6*0.02/0.1 = 1.6e-6 — the two published
# derivation paths must agree exactly, which pins the Fanning/Darcy factors.
# ---------------------------------------------------------------------------

def test_internal_laminar_hagen_poiseuille_two_paths_agree():
    estimate = wallspacing.estimate_wall_spacing(
        velocity=0.02, characteristic_length=0.1,
        kinematic_viscosity=1.0e-6, target_y_plus=1.0,
        flow_type="internal", expansion_ratio=1.2,
    )

    assert estimate.regime == "laminar"
    assert estimate.skin_friction_coefficient.value == pytest.approx(0.008)
    assert "Hagen-Poiseuille" in estimate.skin_friction_coefficient.formula
    # the independent exact path: tau_w/rho = 8*nu*U/D
    assert estimate.kinematic_wall_shear_stress.value == pytest.approx(
        8.0 * 1.0e-6 * 0.02 / 0.1)


# ---------------------------------------------------------------------------
# Property: the spacing scales linearly with the y+ target (both flow types)
# ---------------------------------------------------------------------------

def _spacing(flow_type, y_plus):
    return wallspacing.estimate_wall_spacing(
        velocity=10.0, characteristic_length=1.0,
        kinematic_viscosity=1.0e-6, target_y_plus=y_plus,
        flow_type=flow_type,
    )


@pytest.mark.parametrize("flow_type", ["external", "internal"])
def test_spacing_scales_linearly_with_y_plus_target(flow_type):
    # y1 = y+*nu/u_tau and u_tau does not depend on y+, so a 50x y+ target
    # is exactly 50x the spacing — for the centre AND the height.
    at_one = _spacing(flow_type, 1.0)
    at_fifty = _spacing(flow_type, 50.0)

    assert at_fifty.first_cell_centre_distance.value == pytest.approx(
        50.0 * at_one.first_cell_centre_distance.value)
    assert at_fifty.first_cell_height.value == pytest.approx(
        50.0 * at_one.first_cell_height.value)


@pytest.mark.parametrize("flow_type", ["external", "internal"])
def test_height_is_exactly_twice_the_centre_distance(flow_type):
    # The documented factor (FIRST_CELL_HEIGHT_FACTOR = 2: cell-centred FV,
    # the centre sits at half the cell height) — exactly, not approximately.
    estimate = _spacing(flow_type, 5.0)

    assert wallspacing.FIRST_CELL_HEIGHT_FACTOR == 2.0
    assert estimate.first_cell_height.value == pytest.approx(
        2.0 * estimate.first_cell_centre_distance.value)


# ---------------------------------------------------------------------------
# Property: the regime verdict flips at the documented Reynolds thresholds.
# U = 1, L = 1, so nu = 1/Re puts Re exactly where each case says.
# ---------------------------------------------------------------------------

def _regime_at(flow_type, reynolds):
    return wallspacing.estimate_wall_spacing(
        velocity=1.0, characteristic_length=1.0,
        kinematic_viscosity=1.0 / reynolds, target_y_plus=1.0,
        flow_type=flow_type,
    ).regime


@pytest.mark.parametrize("reynolds,expected", [
    (4.99e5, "laminar"),        # just below the 5e5 transition onset
    (5.0e5, "transitional"),    # at the onset
    (2.99e6, "transitional"),   # just below fully turbulent
    (3.0e6, "turbulent"),       # at the 3e6 fully-turbulent threshold
])
def test_external_regime_flips_at_documented_thresholds(reynolds, expected):
    assert _regime_at("external", reynolds) == expected


@pytest.mark.parametrize("reynolds,expected", [
    (2299.0, "laminar"),        # just below the 2300 laminar ceiling
    (2300.0, "transitional"),   # at it
    (3999.0, "transitional"),   # just below the 4000 turbulent floor
    (4000.0, "turbulent"),      # at it
])
def test_internal_regime_flips_at_documented_thresholds(reynolds, expected):
    assert _regime_at("internal", reynolds) == expected


# ---------------------------------------------------------------------------
# Property: the suggested layer stack actually covers the boundary layer —
# and is the smallest one that does
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flow_type,ratio", [
    ("external", 1.2), ("internal", 1.2),
    ("external", 1.35), ("internal", 1.05),
])
def test_suggested_layer_stack_covers_the_boundary_layer_minimally(
        flow_type, ratio):
    estimate = wallspacing.estimate_wall_spacing(
        velocity=10.0, characteristic_length=0.5,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
        flow_type=flow_type, expansion_ratio=ratio,
    )
    h1 = estimate.first_cell_height.value
    delta = estimate.boundary_layer_thickness.value
    n = estimate.suggested_layer_count.value

    def stack(k):
        return h1 * (ratio ** k - 1.0) / (ratio - 1.0)

    assert stack(n) >= delta                    # covers the boundary layer
    assert n == 1 or stack(n - 1) < delta       # ... and N-1 does not


def test_expansion_ratio_one_gives_uniform_layers():
    estimate = wallspacing.estimate_wall_spacing(
        velocity=10.0, characteristic_length=0.5,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
        flow_type="external", expansion_ratio=1.0,
    )
    h1 = estimate.first_cell_height.value
    delta = estimate.boundary_layer_thickness.value
    n = estimate.suggested_layer_count.value

    assert n * h1 >= delta
    assert (n - 1) * h1 < delta


# ---------------------------------------------------------------------------
# Documented defaults and validity-window honesty
# ---------------------------------------------------------------------------

def test_defaults_are_external_flow_at_snappy_expansion_ratio():
    # flow_type defaults to external; expansion ratio to the snappy
    # reference's conservative 1.2 (documented default, echoed in the result).
    estimate = wallspacing.estimate_wall_spacing(
        velocity=50.0, characteristic_length=1.0,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
    )

    assert wallspacing.DEFAULT_EXPANSION_RATIO == 1.2
    assert estimate.flow_type == "external"
    assert estimate.expansion_ratio == 1.2
    assert estimate.suggested_layer_count.value == \
        _external_turbulent().suggested_layer_count.value


def test_no_extrapolation_note_on_the_validity_boundary():
    # Re_D sitting exactly on the Blasius window's edges (4000 and 1e5) is
    # not an extrapolation — and neither is the flagship water case, whose
    # Re_D lands at 1e5 + 1e-11 of floating-point dust: the evidence stays
    # quiet about validity in all three.
    at_lower_edge = wallspacing.estimate_wall_spacing(
        velocity=4000.0, characteristic_length=1.0,
        kinematic_viscosity=1.0, target_y_plus=30.0,
        flow_type="internal")                       # Re_D = 4000.0 exactly
    at_upper_edge = wallspacing.estimate_wall_spacing(
        velocity=100000.0, characteristic_length=1.0,
        kinematic_viscosity=1.0, target_y_plus=30.0,
        flow_type="internal")                       # Re_D = 1e5 exactly
    for estimate in (at_lower_edge, at_upper_edge, _internal_turbulent()):
        assert "extrapolation" not in "\n".join(estimate.evidence)


def test_blasius_pipe_extrapolation_is_named_in_evidence():
    # Re_D = 1e6 sits above the Blasius pipe correlation's stated validity
    # (4000 < Re_D < 1e5): the number still comes back, flagged as an
    # extrapolation — honesty over silence.
    estimate = wallspacing.estimate_wall_spacing(
        velocity=10.0, characteristic_length=0.1,
        kinematic_viscosity=1.0e-6, target_y_plus=30.0,
        flow_type="internal",
    )

    assert estimate.reynolds_number.value == pytest.approx(1.0e6)
    assert estimate.skin_friction_coefficient.value > 0
    evidence = "\n".join(estimate.evidence)
    assert "extrapolation" in evidence
    assert "validity" in evidence


def test_transitional_regime_names_the_conservative_correlation_choice():
    # Re_x = 1e6 is in the external transitional band (5e5..3e6): sized with
    # the turbulent (Schlichting) correlation, and the evidence says so.
    estimate = wallspacing.estimate_wall_spacing(
        velocity=15.0, characteristic_length=1.0,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
        flow_type="external",
    )

    assert estimate.regime == "transitional"
    assert "Schlichting" in estimate.skin_friction_coefficient.formula
    evidence = "\n".join(estimate.evidence)
    assert "transitional" in evidence
    assert "conservative" in evidence


# ---------------------------------------------------------------------------
# Typed errors: non-physical inputs fail loudly, naming the offending
# parameter — never plausible garbage numbers
# ---------------------------------------------------------------------------

def _estimate(**overrides):
    inputs = dict(
        velocity=10.0, characteristic_length=1.0,
        kinematic_viscosity=1.5e-5, target_y_plus=1.0,
    )
    inputs.update(overrides)
    return wallspacing.estimate_wall_spacing(**inputs)


@pytest.mark.parametrize("name,value", [
    ("velocity", 0.0),
    ("velocity", -10.0),
    ("characteristic_length", 0.0),
    ("characteristic_length", -1.0),
    ("kinematic_viscosity", 0.0),
    ("kinematic_viscosity", -1.5e-5),
    ("target_y_plus", 0.0),
    ("target_y_plus", -30.0),
    ("velocity", float("nan")),
    ("kinematic_viscosity", float("inf")),
])
def test_non_positive_or_non_finite_inputs_raise_naming_the_parameter(
        name, value):
    with pytest.raises(wallspacing.WallSpacingError, match=name):
        _estimate(**{name: value})


def test_unknown_flow_type_raises_naming_the_valid_choices():
    with pytest.raises(wallspacing.WallSpacingError, match="external"):
        _estimate(flow_type="supersonic")


@pytest.mark.parametrize("ratio", [0.0, -1.2, 0.9])
def test_expansion_ratio_below_one_raises(ratio):
    # Layers grow away from the wall: a shrinking (or degenerate) stack can
    # never cover the boundary layer.
    with pytest.raises(wallspacing.WallSpacingError, match="expansion_ratio"):
        _estimate(expansion_ratio=ratio)
