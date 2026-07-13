# turbinlet.py
"""Inlet turbulence calculator: k/epsilon/omega computed, not recalled (issue #68).

``estimate_turbulence_inlet`` is a PURE function — scalars in, typed estimate
out. No filesystem, no subprocess, no case directory: the second purely
computational tool in the server (spec #65). It turns a mean inlet velocity,
a turbulence intensity and a turbulence length scale into the standard
inlet quantities, each carrying the name of the formula that produced it
(evidence style, like assess_mesh/inspect_stl):

    k       = 3/2*(U*I)^2                      [m^2/s^2]
    epsilon = C_mu^(3/4)*k^(3/2)/l             [m^2/s^3]
    omega   = sqrt(k)/(C_mu^(1/4)*l)           [1/s]
    nu_t    = C_mu*k^2/epsilon                 [m^2/s]

plus, when the caller supplies a kinematic viscosity, the turbulent-viscosity
ratio nu_t/nu as the sanity figure — pathological velocity/intensity/length
combinations are visible at a glance (a healthy RAS inlet sits well under
O(1000)). Without nu the ratio is omitted (None), never guessed: the fluid is
the caller's fact, not the tool's assumption.

Silence produces stated assumptions, not hidden ones:

- An omitted intensity applies ``DEFAULT_INTENSITY`` (0.05 — the documented
  medium-turbulence default: between the ~1% of clean external freestreams
  and the >=10% of highly turbulent machinery) and echoes it in the output
  as an applied assumption.
- Exactly ONE of turbulence length scale / hydraulic diameter must be given.
  A hydraulic diameter is converted via the standard mixing-length rule
  ``l = 0.07*D_h`` (``HYDRAULIC_DIAMETER_FACTOR``), with the conversion named
  in the output. Supplying neither or both raises the typed error —
  ambiguity is surfaced, never silently resolved.

Model constants are pinned server-side HERE, in this one place (``C_MU`` and
friends below): the turbulence reference cites this module as the constants'
source rather than restating values, and every output field's formula string
echoes the constant where it enters — the tools and the reference can never
drift apart on a constant.

Non-physical inputs (non-positive or non-finite velocity, intensity, length
scale, hydraulic diameter or viscosity) raise :class:`TurbulenceInletError` —
garbage inputs fail loudly instead of producing plausible garbage numbers.

Like the rest of the mechanical layer this module is key-free and
stdlib-only: CI runs the unit suite with nothing but pytest installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Model constants — pinned server-side in this ONE place (spec #65). The
# turbulence reference cites this module as the source; the formula strings
# below echo each constant where it enters, so tool output and reference can
# never drift apart on a value.
# ---------------------------------------------------------------------------

# The standard k-epsilon model constant (Launder & Spalding 1974), shared by
# the k-omega conversion and the eddy-viscosity relation.
C_MU = 0.09

# Documented medium-turbulence default intensity (5%), applied when the
# caller gives none and echoed in the output as a stated assumption. This is
# the standard medium figure — the same one the Foundation v10 pitzDaily
# tutorial's inlet k derives from (k = 3/2*(10*0.05)^2 = 0.375).
DEFAULT_INTENSITY = 0.05

# The standard mixing-length estimate from a hydraulic diameter, l = 0.07*D_h
# (fully developed duct/pipe turbulence), named in the output whenever the
# conversion is applied.
HYDRAULIC_DIAMETER_FACTOR = 0.07

# Formula names carried by each output field (evidence style): the constant
# is interpolated from the pin above so the echo cannot drift from it.
K_FORMULA = "k = 3/2*(U*I)^2"
EPSILON_FORMULA = f"epsilon = C_mu^(3/4)*k^(3/2)/l (C_mu = {C_MU})"
OMEGA_FORMULA = f"omega = sqrt(k)/(C_mu^(1/4)*l) (C_mu = {C_MU})"
NU_T_FORMULA = f"nu_t = C_mu*k^2/epsilon (C_mu = {C_MU})"
VISCOSITY_RATIO_FORMULA = "viscosity_ratio = nu_t/nu"
DIAMETER_RULE = f"l = {HYDRAULIC_DIAMETER_FACTOR}*D_h"


class TurbulenceInletError(RuntimeError):
    """The inlet estimate cannot be computed from the given inputs
    (non-physical value, or neither/both of turbulence length scale and
    hydraulic diameter). Never plausible garbage numbers."""


@dataclass
class InletQuantity:
    """One computed inlet quantity carrying the name of the formula that
    produced it (with the pinned constant echoed where it enters)."""
    value: float
    units: str
    formula: str


@dataclass
class TurbulenceInletEstimate:
    """Typed inlet turbulence estimate: the applied inputs with their
    provenance (caller-supplied vs documented default/conversion), the
    pinned model constant, and each quantity with its formula name."""
    velocity: float               # U [m/s], echoed input
    intensity: float              # I [-] as applied (fraction, 0.05 = 5%)
    intensity_source: str         # "caller-supplied" or the named default
    length_scale: float           # l [m] as applied
    length_scale_source: str      # "caller-supplied" or the named conversion
    c_mu: float                   # the pinned model constant, echoed
    k: InletQuantity
    epsilon: InletQuantity
    omega: InletQuantity
    nu_t: InletQuantity
    viscosity_ratio: InletQuantity | None   # nu_t/nu when nu is given
    assumptions: list[str] = field(default_factory=list)


def _require_physical(name: str, value: float) -> None:
    """A named input must be a finite number greater than zero."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or not math.isfinite(value) or value <= 0:
        raise TurbulenceInletError(
            f"{name} must be a finite number greater than zero, got {value!r} "
            "— refusing to compute plausible garbage from a non-physical input."
        )


def estimate_turbulence_inlet(
    velocity: float,
    intensity: float | None = None,
    length_scale: float | None = None,
    hydraulic_diameter: float | None = None,
    kinematic_viscosity: float | None = None,
) -> TurbulenceInletEstimate:
    """Compute inlet k/epsilon/omega/nu_t from velocity, intensity and a
    length scale. Pure function (module docstring has the formulas).

    ``intensity`` is a fraction (0.05 = 5%); omitted, the documented medium
    default ``DEFAULT_INTENSITY`` is applied and echoed as an assumption.
    Exactly one of ``length_scale`` [m] / ``hydraulic_diameter`` [m] must be
    given (the diameter converts via the named ``l = 0.07*D_h`` rule);
    neither or both raises :class:`TurbulenceInletError`, as does any
    non-positive or non-finite input. With ``kinematic_viscosity`` [m^2/s]
    the turbulent-viscosity ratio nu_t/nu is included as the sanity figure;
    without it the ratio is None — the fluid is never assumed.
    """
    if (length_scale is None) == (hydraulic_diameter is None):
        given = ("both" if length_scale is not None else "neither")
        raise TurbulenceInletError(
            "supply exactly one of length_scale / hydraulic_diameter "
            f"(got {given}) — the length scale would be ambiguous otherwise."
        )

    _require_physical("velocity", velocity)
    if intensity is not None:
        _require_physical("intensity", intensity)
    if length_scale is not None:
        _require_physical("length_scale", length_scale)
    if hydraulic_diameter is not None:
        _require_physical("hydraulic_diameter", hydraulic_diameter)
    if kinematic_viscosity is not None:
        _require_physical("kinematic_viscosity", kinematic_viscosity)

    assumptions: list[str] = []

    if intensity is None:
        intensity = DEFAULT_INTENSITY
        intensity_source = (
            f"documented medium-turbulence default (I = {DEFAULT_INTENSITY})"
        )
        assumptions.append(
            "turbulence intensity not given — applied the documented "
            f"medium-turbulence default I = {DEFAULT_INTENSITY} "
            f"({DEFAULT_INTENSITY:.0%})."
        )
    else:
        intensity_source = "caller-supplied"

    if length_scale is None:
        length_scale = HYDRAULIC_DIAMETER_FACTOR * hydraulic_diameter
        length_scale_source = (
            f"{DIAMETER_RULE} from hydraulic diameter "
            f"D_h = {hydraulic_diameter:.6g} m"
        )
        assumptions.append(
            "turbulence length scale derived from the hydraulic diameter "
            f"via the standard mixing-length rule {DIAMETER_RULE}: "
            f"l = {length_scale:.6g} m."
        )
    else:
        length_scale_source = "caller-supplied"

    k = 1.5 * (velocity * intensity) ** 2
    epsilon = C_MU ** 0.75 * k ** 1.5 / length_scale
    omega = math.sqrt(k) / (C_MU ** 0.25 * length_scale)
    nu_t = C_MU * k ** 2 / epsilon

    viscosity_ratio = None
    if kinematic_viscosity is not None:
        viscosity_ratio = InletQuantity(
            value=nu_t / kinematic_viscosity, units="-",
            formula=VISCOSITY_RATIO_FORMULA,
        )

    return TurbulenceInletEstimate(
        velocity=velocity,
        intensity=intensity,
        intensity_source=intensity_source,
        length_scale=length_scale,
        length_scale_source=length_scale_source,
        c_mu=C_MU,
        k=InletQuantity(value=k, units="m^2/s^2", formula=K_FORMULA),
        epsilon=InletQuantity(value=epsilon, units="m^2/s^3",
                              formula=EPSILON_FORMULA),
        omega=InletQuantity(value=omega, units="1/s", formula=OMEGA_FORMULA),
        nu_t=InletQuantity(value=nu_t, units="m^2/s", formula=NU_T_FORMULA),
        viscosity_ratio=viscosity_ratio,
        assumptions=assumptions,
    )
