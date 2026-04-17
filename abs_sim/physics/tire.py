"""Dugoff combined-slip tire model and surface presets.

Dugoff tire model computes longitudinal (Fx) and lateral (Fy) forces from
slip ratio (kappa), slip angle (alpha), normal load (Fz), and tire/road
friction (mu), using a friction-ellipse saturation. It takes four parameters
(Cx, Cy, mu, Fz) and is standard in academic ABS / control-design work.

References
----------
* Dugoff, H., Fancher, P., and Segel, L., 1970, "An analysis of tire traction
  properties and their influence on vehicle dynamic performance".
* Brown, A. "simply python/numpy Dugoff tire model"
  https://gist.github.com/Alexanderallenbrown/e315b52a32dbebece7f1
* Pacejka, H.B., "Tire and Vehicle Dynamics" (2012), 3rd ed., Ch. 4.
* MathWorks "Tire-Road Interaction (Magic Formula)" surface coefficients.

Sign conventions (SAE tire axis, right-handed x-forward y-right z-down)
----------------------------------------------------------------------
* kappa = (omega * R_eff - vx_tire) / max(|vx_tire|, eps),
  positive under driving, negative under braking.
* alpha = atan2(-vy_tire, |vx_tire|),
  positive when the velocity vector lies to the right of the wheel heading.
* Fx > 0: tire pushes vehicle forward; Fx < 0: tire brakes the vehicle.
* Fy > 0: tire pushes to tire-right (positive y in tire frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math


EPS = 1e-6


# --------------------------------------------------------------------------- #
# Surface presets
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Surface:
    """A road surface with a friction coefficient and a display color."""

    name: str
    mu: float
    color: Tuple[int, int, int]


SURFACES: Dict[str, Surface] = {
    "dry":  Surface(name="dry",  mu=0.90, color=(80, 80, 80)),
    "wet":  Surface(name="wet",  mu=0.60, color=(60, 90, 150)),
    "snow": Surface(name="snow", mu=0.30, color=(215, 225, 235)),
    "ice":  Surface(name="ice",  mu=0.10, color=(170, 210, 230)),
    "sand": Surface(name="sand", mu=0.35, color=(200, 170, 110)),
}


def get_surface(name: str) -> Surface:
    """Look up a surface by name, case-insensitive. KeyError if unknown."""
    key = name.lower()
    if key not in SURFACES:
        raise KeyError(f"Unknown surface '{name}'. Known: {list(SURFACES)}")
    return SURFACES[key]


# --------------------------------------------------------------------------- #
# Dugoff tire model
# --------------------------------------------------------------------------- #

@dataclass
class DugoffTire:
    """Dugoff combined-slip tire model with post-peak friction fade.

    Parameters
    ----------
    Cx           : longitudinal stiffness (N per unit slip ratio). Typical 40k-200k.
    Cy           : cornering stiffness (N per radian of slip angle). Typical 40k-120k.
    R            : effective rolling radius (m).
    slip_peak    : combined-slip magnitude at which friction peaks (rad / unit).
                   Above this the slide-friction fade engages.
    slip_slide   : combined-slip magnitude at which friction has fully faded
                   to mu * slide_ratio.
    slide_ratio  : mu_slide / mu_peak. Real rubber tires typically 0.75-0.95.
                   A lower value is what makes ABS actually beneficial: locked
                   wheels lose grip relative to a wheel controlled near peak.

    The baseline Dugoff gives a sharp knee and plateau at mu*Fz; real tires
    have a distinct peak at a finite slip (kappa ~= 0.15) and a sliding value
    lower than the peak. We model this by first computing the standard
    saturated Dugoff force and then multiplying by a slip-dependent fade
    factor that goes from 1.0 at slip_peak down to slide_ratio at slip_slide.
    """

    Cx: float = 80000.0
    Cy: float = 60000.0
    R: float = 0.32
    slip_peak: float = 0.15
    slip_slide: float = 0.5
    slide_ratio: float = 0.75

    def _fade(self, slip_mag: float) -> float:
        """Post-peak slide-friction fade factor in [slide_ratio, 1]."""
        if slip_mag <= self.slip_peak:
            return 1.0
        if slip_mag >= self.slip_slide:
            return self.slide_ratio
        t = (slip_mag - self.slip_peak) / max(self.slip_slide - self.slip_peak, EPS)
        return 1.0 - (1.0 - self.slide_ratio) * t

    def forces(
        self,
        kappa: float,
        alpha: float,
        Fz: float,
        mu: float,
    ) -> Tuple[float, float]:
        """Return (Fx, Fy) in the tire frame, in newtons.

        1. Linear tire forces: Fxl = Cx*kappa, Fyl = Cy*tan(alpha).
        2. Dugoff saturation so |F| asymptotes to mu*Fz at large slip.
        3. Slide-friction fade post-peak so the force actually DROPS beyond
           the peak, which is what lets ABS outperform a locked wheel.
        """
        if Fz <= 0.0:
            return 0.0, 0.0

        Fx_lin = self.Cx * kappa
        Fy_lin = self.Cy * math.tan(alpha)
        Fmag = math.hypot(Fx_lin, Fy_lin)
        Fmax = mu * Fz

        if Fmag < EPS or Fmax <= 0.0:
            return 0.0, 0.0

        lam = Fmax / (2.0 * Fmag)
        if lam < 1.0:
            scale = lam * (2.0 - lam)
        else:
            scale = 1.0

        Fx = Fx_lin * scale
        Fy = Fy_lin * scale

        slip_mag = math.hypot(kappa, math.tan(alpha))
        fade = self._fade(slip_mag)
        return Fx * fade, Fy * fade

    def pure_longitudinal(self, kappa: float, Fz: float, mu: float) -> float:
        """Convenience: Fx for pure longitudinal slip (alpha = 0)."""
        return self.forces(kappa, 0.0, Fz, mu)[0]

    def pure_lateral(self, alpha: float, Fz: float, mu: float) -> float:
        """Convenience: Fy for pure lateral slip (kappa = 0)."""
        return self.forces(0.0, alpha, Fz, mu)[1]


def slip_risk(kappa: float, kappa_opt: float = 0.15) -> float:
    """Normalize slip-distance-from-peak to [0, 1] for inter-subsystem signaling.

    0 means the wheel is rolling happily at or below peak-grip slip; 1 means
    the wheel is locked (|kappa| >= 1). Anywhere in between scales linearly
    over the post-peak zone.
    """
    a = abs(kappa)
    if a <= kappa_opt:
        return 0.0
    if a >= 1.0:
        return 1.0
    return (a - kappa_opt) / (1.0 - kappa_opt)
