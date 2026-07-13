# wallspacing.py
"""Wall-spacing calculator: first-cell height from flow conditions and a
target y+ (issue #67).

``estimate_wall_spacing(velocity, characteristic_length, kinematic_viscosity,
target_y_plus, flow_type, expansion_ratio)`` is a PURE function — the first
purely computational capability in the server: no case directory, no
filesystem, no subprocess, no sourced environment. It turns flow conditions
and a target y+ into the numbers a boundary-layer mesh needs, each paired
with the name of the correlation or formula that produced it: Reynolds
number with a regime verdict, skin-friction coefficient, kinematic wall
shear stress, friction velocity, the first-cell-CENTRE distance and the
first-cell HEIGHT as two separately labelled fields (the classic factor-of-2
confusion dies in the schema), a boundary-layer thickness estimate, and a
suggested layer count covering that thickness at the given expansion ratio.
Numbers are computed, never recalled.

Pinned correlations (the module's single source of truth; each name is
echoed in the output field it produced; sources verified 2026-07-13 against
the published pages/texts named below):

- External (flat-plate family), Re_x = U*L/nu:
  - laminar: Blasius similarity solution (exact),
    Cf = 0.664*Re_x^(-1/2); delta = 5.0*L/sqrt(Re_x)
    (White, Fluid Mechanics; Incropera ch. 7).
  - transitional/turbulent: Schlichting flat-plate local skin friction,
    Cf = (2*log10(Re_x) - 0.65)^(-2.3), valid Re_x < 1e9 (Schlichting,
    Boundary-Layer Theory; the correlation used by the standard y+
    calculators — Quadco Engineering, Flowthermolab);
    delta = 0.37*L*Re_x^(-1/5) (1/7th-power-law growth, Schlichting).
- Internal (pipe family), Re_D = U*D_h/nu; Cf is FANNING (tau_w = Cf/2*rho*U^2),
  converted from the Darcy factor f as Cf = f/4:
  - laminar: Hagen-Poiseuille fully developed flow (exact), f = 64/Re_D.
  - transitional/turbulent: Blasius smooth pipe, f = 0.316*Re_D^(-1/4),
    stated validity 4000 < Re_D < 1e5 (Flowthermolab page; the LEAP CFD
    page prints the identical correlation in Fanning form 0.079*Re^(-0.25)).
    Outside the stated validity the value is an extrapolation and the
    evidence says so — flagged, never silently trusted.
- The spacing chain (identical on every published source):
  tau_w/rho = (Cf/2)*U^2; u_tau = sqrt(tau_w/rho) = U*sqrt(Cf/2);
  y1 = y+*nu/u_tau. The tool takes kinematic viscosity only, so the wall
  shear stress is reported KINEMATIC (tau_w/rho, m^2/s^2) — density cancels
  everywhere else in the chain and is never needed.

Documented mechanical defaults (per-application judgement is skill-side):

- Regime thresholds. External flat plate: laminar below Re_x = 5e5
  (transition onset — Incropera sec. 7.2), fully turbulent at/above
  Re_x = 3e6 (White, Fluid Mechanics: transition band 5e5..3e6),
  transitional between. Internal pipe: laminar below Re_D = 2300, turbulent
  at/above Re_D = 4000, transitional between (Incropera sec. 8.1; Cengel).
- The transitional band is sized with the TURBULENT correlation — the
  conservative choice for meshing (overpredicts friction, undersizes the
  first cell); the evidence names it.
- A laminar-regime result still returns every number (from the laminar
  correlation); the regime verdict tells the caller that wall functions and
  turbulence models are inapplicable.
- First-cell HEIGHT = FIRST_CELL_HEIGHT_FACTOR (2) x the first-cell-CENTRE
  distance: OpenFOAM is cell-centred, so the wall-adjacent cell's centre —
  where y+ is evaluated — sits at half the cell height.
- Internal boundary-layer thickness is D_h/2: fully developed pipe flow's
  wall layers meet at the axis, so the layer stack aims at the radius.
- Suggested layer count: the smallest N whose geometric stack
  h1*(r^N - 1)/(r - 1) covers delta (N = ceil(delta/h1) when r == 1),
  floored at 1. The default expansion ratio DEFAULT_EXPANSION_RATIO (1.2)
  is the snappy reference's conservative default
  (agents/skills/foam/references/snappyhexmesh.md).

Typed errors: non-positive or non-finite velocity, length, viscosity or y+
target, an expansion ratio below 1, or an unknown flow type raise
:class:`WallSpacingError` — garbage inputs fail loudly instead of producing
plausible garbage numbers.

Like the rest of the mechanical layer this module is key-free and
stdlib-only (CI runs the unit suite with nothing but pytest installed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Documented defaults (see module docstring).
DEFAULT_EXPANSION_RATIO = 1.2   # the snappy reference's conservative default
FIRST_CELL_HEIGHT_FACTOR = 2.0  # cell height = 2 x centre distance (cell-centred FV)

# Regime thresholds (module docstring names the sources).
RE_X_TRANSITION_ONSET = 5.0e5   # external: laminar below this
RE_X_FULLY_TURBULENT = 3.0e6    # external: turbulent at/above this
RE_D_LAMINAR_MAX = 2300.0       # internal: laminar below this
RE_D_TURBULENT_MIN = 4000.0     # internal: turbulent at/above this

# Stated validity windows of the pinned turbulent correlations; outside them
# the value is an extrapolation and the evidence says so.
SCHLICHTING_RE_MAX = 1.0e9              # Cf = (2*log10 Re - 0.65)^-2.3
BLASIUS_PIPE_RE_MIN = 4000.0            # f = 0.316*Re^-0.25 ...
BLASIUS_PIPE_RE_MAX = 1.0e5             # ... stated 4000 < Re_D < 1e5

FLOW_TYPES = ("external", "internal")


class WallSpacingError(ValueError):
    """Non-physical wall-spacing input (non-positive/non-finite velocity,
    length, viscosity or y+ target; expansion ratio below 1; unknown flow
    type). Fails loudly — never plausible garbage numbers."""


@dataclass
class Quantity:
    """One computed number paired with the name of the correlation/formula
    that produced it (evidence style: numbers never travel nameless)."""
    value: float
    formula: str


@dataclass
class LayerCount:
    """The suggested prism-layer count with the covering rule that chose it."""
    value: int
    formula: str


@dataclass
class WallSpacingEstimate:
    """Typed wall-spacing estimate: every numeric field carries its formula."""
    flow_type: str                  # "external" or "internal"
    velocity: float                 # U [m/s] (echoed input)
    characteristic_length: float    # L / D_h [m] (echoed input)
    kinematic_viscosity: float      # nu [m^2/s] (echoed input)
    target_y_plus: float            # y+ target (echoed input)
    expansion_ratio: float          # layer growth ratio r (echoed input)
    reynolds_number: Quantity
    regime: str                     # "laminar", "transitional" or "turbulent"
    skin_friction_coefficient: Quantity
    kinematic_wall_shear_stress: Quantity   # tau_w/rho [m^2/s^2]
    friction_velocity: Quantity             # u_tau [m/s]
    first_cell_centre_distance: Quantity    # y1 [m] — wall-adjacent cell CENTRE
    first_cell_height: Quantity             # h1 = 2*y1 [m] — cell HEIGHT
    boundary_layer_thickness: Quantity      # delta [m]
    suggested_layer_count: LayerCount
    evidence: list[str] = field(default_factory=list)


def _require_positive(name: str, value: float) -> float:
    """A finite, strictly positive number — or the typed error naming the
    offending parameter (never plausible garbage numbers)."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise WallSpacingError(f"{name} must be a number (got {value!r})") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise WallSpacingError(
            f"{name} must be a positive finite number (got {value:g})"
        )
    return value


def _regime(flow_type: str, reynolds: float) -> tuple[str, str]:
    """The regime verdict plus the evidence line naming the thresholds."""
    if flow_type == "external":
        thresholds = (
            f"external flat-plate thresholds on Re_x: laminar below "
            f"{RE_X_TRANSITION_ONSET:g} (transition onset), turbulent "
            f"at/above {RE_X_FULLY_TURBULENT:g}, transitional between"
        )
        if reynolds < RE_X_TRANSITION_ONSET:
            regime = "laminar"
        elif reynolds < RE_X_FULLY_TURBULENT:
            regime = "transitional"
        else:
            regime = "turbulent"
    else:
        thresholds = (
            f"internal pipe thresholds on Re_D: laminar below "
            f"{RE_D_LAMINAR_MAX:g}, turbulent at/above "
            f"{RE_D_TURBULENT_MIN:g}, transitional between"
        )
        if reynolds < RE_D_LAMINAR_MAX:
            regime = "laminar"
        elif reynolds < RE_D_TURBULENT_MIN:
            regime = "transitional"
        else:
            regime = "turbulent"
    return regime, f"Re = {reynolds:.6g}: {regime} ({thresholds})"


def _skin_friction(flow_type: str, regime: str,
                   reynolds: float) -> tuple[Quantity, list[str]]:
    """The pinned skin-friction correlation for the flow type and regime:
    the Fanning-convention Cf (tau_w = Cf/2*rho*U^2) with its name, plus any
    extrapolation evidence when Re sits outside the stated validity."""
    notes: list[str] = []
    if flow_type == "external":
        if regime == "laminar":
            cf = 0.664 * reynolds ** -0.5
            formula = ("Blasius laminar flat plate (exact): "
                       "Cf = 0.664*Re_x^(-1/2)")
        else:
            cf = (2.0 * math.log10(reynolds) - 0.65) ** -2.3
            formula = ("Schlichting flat-plate local skin friction: "
                       "Cf = (2*log10(Re_x) - 0.65)^(-2.3), "
                       "valid Re_x < 1e9")
            if reynolds >= SCHLICHTING_RE_MAX:
                notes.append(
                    f"Re_x = {reynolds:.6g} is at/above the Schlichting "
                    f"correlation's stated validity ({SCHLICHTING_RE_MAX:g}) "
                    "— the skin friction is an extrapolation"
                )
    else:
        if regime == "laminar":
            f_darcy = 64.0 / reynolds
            formula = ("Hagen-Poiseuille fully developed pipe (exact): "
                       "Cf = f/4, f = 64/Re_D (Fanning convention)")
        else:
            f_darcy = 0.316 * reynolds ** -0.25
            formula = ("Blasius smooth pipe: Cf = f/4, "
                       "f = 0.316*Re_D^(-1/4), valid 4000 < Re_D < 1e5 "
                       "(Fanning convention)")
            if not (BLASIUS_PIPE_RE_MIN < reynolds <= BLASIUS_PIPE_RE_MAX):
                notes.append(
                    f"Re_D = {reynolds:.6g} is outside the Blasius pipe "
                    f"correlation's stated validity "
                    f"({BLASIUS_PIPE_RE_MIN:g} < Re_D < "
                    f"{BLASIUS_PIPE_RE_MAX:g}) — the skin friction is an "
                    "extrapolation"
                )
        cf = f_darcy / 4.0
    return Quantity(value=cf, formula=formula), notes


def _boundary_layer_thickness(flow_type: str, regime: str, reynolds: float,
                              length: float) -> Quantity:
    """The pinned boundary-layer thickness estimate for the flow type and
    regime (module docstring names the sources)."""
    if flow_type == "external":
        if regime == "laminar":
            return Quantity(
                value=5.0 * length * reynolds ** -0.5,
                formula="Blasius laminar flat plate: delta = 5.0*L/sqrt(Re_x)",
            )
        return Quantity(
            value=0.37 * length * reynolds ** -0.2,
            formula=("1/7th-power-law turbulent flat plate: "
                     "delta = 0.37*L*Re_x^(-1/5)"),
        )
    return Quantity(
        value=length / 2.0,
        formula=("fully developed pipe flow: delta = D_h/2 "
                 "(wall layers meet at the axis)"),
    )


def _layer_count(first_cell_height: float, thickness: float,
                 ratio: float) -> LayerCount:
    """The smallest geometric layer stack (first layer = the first-cell
    height, growth = ratio) that covers the boundary-layer thickness."""
    if ratio == 1.0:
        count = max(1, math.ceil(thickness / first_cell_height))
        formula = ("N = ceil(delta/h1), uniform layers (expansion ratio 1) "
                   "covering the boundary-layer thickness")
    else:
        count = max(1, math.ceil(
            math.log(1.0 + thickness * (ratio - 1.0) / first_cell_height)
            / math.log(ratio)
        ))
        formula = ("smallest N with h1*(r^N - 1)/(r - 1) >= delta "
                   "(geometric layer stack covering the boundary-layer "
                   "thickness)")
    return LayerCount(value=count, formula=formula)


def estimate_wall_spacing(
    velocity: float,
    characteristic_length: float,
    kinematic_viscosity: float,
    target_y_plus: float,
    flow_type: str = "external",
    expansion_ratio: float = DEFAULT_EXPANSION_RATIO,
) -> WallSpacingEstimate:
    """Estimate the wall-normal mesh spacing for a target y+. Pure function.

    Inputs are SI: velocity U [m/s], characteristic length [m] (plate/body
    length for external flow, hydraulic diameter for internal), kinematic
    viscosity nu [m^2/s], the y+ the first cell CENTRE should land on, the
    flow type selecting the correlation family, and the layer expansion
    ratio (default 1.2, the snappy reference's conservative default).

    Returns a :class:`WallSpacingEstimate` where every numeric field carries
    the name of the correlation/formula that produced it. Raises
    :class:`WallSpacingError` on non-physical inputs — never plausible
    garbage numbers.
    """
    velocity = _require_positive("velocity", velocity)
    characteristic_length = _require_positive(
        "characteristic_length", characteristic_length)
    kinematic_viscosity = _require_positive(
        "kinematic_viscosity", kinematic_viscosity)
    target_y_plus = _require_positive("target_y_plus", target_y_plus)
    if flow_type not in FLOW_TYPES:
        raise WallSpacingError(
            f"flow_type must be one of {FLOW_TYPES} (got {flow_type!r})"
        )
    try:
        expansion_ratio = float(expansion_ratio)
    except (TypeError, ValueError) as exc:
        raise WallSpacingError(
            f"expansion_ratio must be a number (got {expansion_ratio!r})"
        ) from exc
    if not math.isfinite(expansion_ratio) or expansion_ratio < 1.0:
        raise WallSpacingError(
            f"expansion_ratio must be at least 1 (got {expansion_ratio:g}) — "
            "boundary layers grow away from the wall"
        )

    reynolds = velocity * characteristic_length / kinematic_viscosity
    re_symbol = "Re_x" if flow_type == "external" else "Re_D"
    length_symbol = "L" if flow_type == "external" else "D_h"
    reynolds_quantity = Quantity(
        value=reynolds, formula=f"{re_symbol} = U*{length_symbol}/nu")

    regime, regime_evidence = _regime(flow_type, reynolds)
    evidence = [regime_evidence]
    if regime == "laminar":
        evidence.append(
            "laminar regime — wall functions and turbulence models are "
            "inapplicable; the numbers below use the laminar correlation"
        )
    elif regime == "transitional":
        evidence.append(
            "transitional regime — sized with the turbulent correlation "
            "(the conservative choice for meshing: it overpredicts friction "
            "and undersizes the first cell)"
        )

    skin_friction, validity_notes = _skin_friction(flow_type, regime, reynolds)
    evidence.extend(validity_notes)

    cf = skin_friction.value
    shear = Quantity(
        value=0.5 * cf * velocity ** 2,
        formula=("tau_w/rho = (Cf/2)*U^2 — KINEMATIC wall shear stress "
                 "[m^2/s^2]; the tool takes nu only, density cancels in the "
                 "spacing chain"),
    )
    friction_velocity = Quantity(
        value=velocity * math.sqrt(cf / 2.0),
        formula="u_tau = U*sqrt(Cf/2) = sqrt(tau_w/rho)",
    )
    centre = Quantity(
        value=target_y_plus * kinematic_viscosity / friction_velocity.value,
        formula=("y1 = y+*nu/u_tau — distance of the first cell CENTRE "
                 "from the wall"),
    )
    height = Quantity(
        value=FIRST_CELL_HEIGHT_FACTOR * centre.value,
        formula=(f"h1 = {FIRST_CELL_HEIGHT_FACTOR:g}*y1 — first-cell HEIGHT "
                 "(cell-centred FV: the centre sits at half the cell height)"),
    )
    thickness = _boundary_layer_thickness(
        flow_type, regime, reynolds, characteristic_length)
    layers = _layer_count(height.value, thickness.value, expansion_ratio)
    evidence.append(
        f"{layers.value} layer(s) at expansion ratio {expansion_ratio:g} "
        f"starting from first-cell height {height.value:.6g} m cover the "
        f"boundary-layer thickness estimate {thickness.value:.6g} m"
    )

    return WallSpacingEstimate(
        flow_type=flow_type,
        velocity=velocity,
        characteristic_length=characteristic_length,
        kinematic_viscosity=kinematic_viscosity,
        target_y_plus=target_y_plus,
        expansion_ratio=expansion_ratio,
        reynolds_number=reynolds_quantity,
        regime=regime,
        skin_friction_coefficient=skin_friction,
        kinematic_wall_shear_stress=shear,
        friction_velocity=friction_velocity,
        first_cell_centre_distance=centre,
        first_cell_height=height,
        boundary_layer_thickness=thickness,
        suggested_layer_count=layers,
        evidence=evidence,
    )
