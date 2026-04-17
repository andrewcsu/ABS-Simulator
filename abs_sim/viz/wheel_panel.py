"""2x2 per-wheel status panel.

For each wheel, draws:
* A rotating disc whose orientation tracks the wheel's spin angle
  (normalized visually so you can SEE that the wheel is spinning or locked).
* A slip-risk bar in [0, 1], green -> yellow -> red.
* A brake-pressure bar that pulses with ABS cycling.
* The surface currently under that wheel, labelled and coloured.
* A normal-load bar showing static + transferred load.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import pygame

from abs_sim.control.wheel_abs import ABSState
from abs_sim.physics.tire import SURFACES, slip_risk
from abs_sim.sim.simulation import Car
from abs_sim.viz.top_down import _get_font


def _slip_color(risk: float) -> Tuple[int, int, int]:
    r = max(0.0, min(1.0, risk))
    if r < 0.5:
        g = 220
        red = int(40 + 215 * (r / 0.5))
    else:
        red = 255
        g = int(220 * (1.0 - (r - 0.5) / 0.5))
    return (red, g, 60)


def _state_color(state: ABSState) -> Tuple[int, int, int]:
    if state == ABSState.APPLY:
        return (60, 180, 110)
    if state == ABSState.HOLD:
        return (230, 200, 80)
    return (220, 70, 70)  # RELEASE


class WheelPanelRenderer:
    def __init__(self, viewport: Tuple[int, int, int, int]) -> None:
        """viewport = (x, y, w, h) in pixels."""
        self.x, self.y, self.w, self.h = viewport
        # Visual spin angle, accumulated. Keeps the disc indicator visibly
        # rotating at real scale without having to expose omega geometry.
        self._visual_spin = [0.0, 0.0, 0.0, 0.0]

    def draw(self, surf: pygame.Surface, car: Car, dt_real: float) -> None:
        kin = car.vehicle.wheel_kinematics()
        states = car.last_abs_states
        pressures = car.last_actuator_pressure

        cell_w = self.w // 2
        cell_h = self.h // 2
        layout = [(0, 0), (1, 0), (0, 1), (1, 1)]  # FL, FR, RL, RR
        labels = ("FL", "FR", "RL", "RR")

        for i, (cx_i, cy_i) in enumerate(layout):
            rect = pygame.Rect(
                self.x + cx_i * cell_w,
                self.y + cy_i * cell_h,
                cell_w,
                cell_h,
            )
            self._draw_cell(
                surf, rect, labels[i],
                omega=car.vehicle.wheel_speeds[i],
                kappa=kin[i].kappa,
                Fz=kin[i].Fz,
                mu=car.last_mu[i],
                state=states[i],
                pressure=pressures[i],
                surface_name=car.last_surface,
                dt_real=dt_real,
                index=i,
            )

    def _draw_cell(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        omega: float,
        kappa: float,
        Fz: float,
        mu: float,
        state: ABSState,
        pressure: float,
        surface_name: str,
        dt_real: float,
        index: int,
    ) -> None:
        pygame.draw.rect(surf, (28, 30, 34), rect)
        pygame.draw.rect(surf, (70, 72, 78), rect, 1)

        pad = 8
        font = _get_font(14)
        font_small = _get_font(12)

        # Label + state letter
        header = font.render(f"{label}", True, (235, 235, 235))
        surf.blit(header, (rect.left + pad, rect.top + pad))
        col = _state_color(state)
        letter = font.render(f"[{state.value}]", True, col)
        surf.blit(letter, (rect.left + pad + header.get_width() + 8, rect.top + pad))

        # Rotating disc
        disc_r = min(cell for cell in (rect.width, rect.height)) // 4
        disc_cx = rect.left + pad + disc_r + 4
        disc_cy = rect.top + pad + header.get_height() + disc_r + 8
        pygame.draw.circle(surf, (90, 90, 98), (disc_cx, disc_cy), disc_r)
        pygame.draw.circle(surf, (200, 200, 210), (disc_cx, disc_cy), disc_r, 2)

        # Animate at clamped rate so lockup is visible
        self._visual_spin[index] += omega * dt_real
        tip_angle = self._visual_spin[index]
        for k in range(6):
            a = tip_angle + k * math.pi / 3.0
            ex = disc_cx + int(disc_r * 0.9 * math.cos(a))
            ey = disc_cy + int(disc_r * 0.9 * math.sin(a))
            pygame.draw.line(surf, (230, 230, 240), (disc_cx, disc_cy), (ex, ey), 1)
        if abs(kappa) > 0.9:
            pygame.draw.circle(surf, (250, 60, 60), (disc_cx, disc_cy), disc_r + 3, 2)

        # Bars column (right side)
        bars_x = disc_cx + disc_r + 16
        bars_w = max(60, rect.right - bars_x - pad)
        bar_h = 10
        bar_y = rect.top + pad + header.get_height() + 4

        # Slip-risk bar [0, 1]
        risk = slip_risk(kappa)
        self._draw_bar(surf, bars_x, bar_y, bars_w, bar_h, risk, _slip_color(risk), "slip")
        bar_y += bar_h + 14

        # Brake pressure bar
        self._draw_bar(surf, bars_x, bar_y, bars_w, bar_h, pressure,
                       _state_color(state), "brake")
        bar_y += bar_h + 14

        # Normal-load bar, normalized around the static load (~3600 N per wheel)
        nom = 3700.0
        self._draw_bar(surf, bars_x, bar_y, bars_w, bar_h,
                       min(Fz / (2 * nom), 1.0),
                       (120, 160, 230), f"load {int(Fz)} N")
        bar_y += bar_h + 16

        # Numeric readout
        readout = font_small.render(
            f"kappa={kappa:+.2f}  mu={mu:.2f}",
            True, (210, 215, 220),
        )
        surf.blit(readout, (rect.left + pad, rect.bottom - readout.get_height() - pad - 16))

        # Surface tag
        surface_col = SURFACES.get(surface_name, SURFACES["dry"]).color
        tag_rect = pygame.Rect(rect.left + pad, rect.bottom - 16 - pad, bars_w + 60, 12)
        pygame.draw.rect(surf, surface_col, tag_rect)
        pygame.draw.rect(surf, (180, 180, 190), tag_rect, 1)
        stext = font_small.render(surface_name, True, (20, 20, 20))
        surf.blit(stext, (tag_rect.left + 4, tag_rect.top - 1))

    def _draw_bar(
        self,
        surf: pygame.Surface,
        x: int, y: int, w: int, h: int,
        value: float,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        value = max(0.0, min(1.0, value))
        pygame.draw.rect(surf, (50, 52, 58), (x, y, w, h))
        pygame.draw.rect(surf, color, (x, y, int(w * value), h))
        pygame.draw.rect(surf, (110, 115, 120), (x, y, w, h), 1)
        f = _get_font(11)
        t = f.render(label, True, (220, 220, 228))
        surf.blit(t, (x, y - t.get_height() - 1))
