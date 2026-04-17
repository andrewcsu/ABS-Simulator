"""Heads-up display: top-line car status, bottom-line event/help bar."""

from __future__ import annotations

from typing import Tuple

import pygame

from abs_sim.sim.simulation import Car
from abs_sim.viz.top_down import _get_font


M_TO_MPH = 2.23694


class HUDRenderer:
    def __init__(self, viewport: Tuple[int, int, int, int]) -> None:
        self.x, self.y, self.w, self.h = viewport

    def draw(
        self,
        surf: pygame.Surface,
        car: Car,
        sim_time: float,
        extra_lines: list = None,
    ) -> None:
        rect = pygame.Rect(self.x, self.y, self.w, self.h)
        pygame.draw.rect(surf, (20, 22, 26), rect)
        pygame.draw.rect(surf, (60, 60, 66), rect, 1)

        pad = 12
        font_big = _get_font(22)
        font = _get_font(16)
        font_small = _get_font(13)

        v = car.vehicle
        mph = v.speed * M_TO_MPH
        big = font_big.render(
            f"{v.speed:5.1f} m/s   ({mph:5.1f} mph)",
            True, (240, 240, 250),
        )
        surf.blit(big, (self.x + pad, self.y + pad))

        line = font.render(
            (
                f"t={sim_time:5.2f}s  "
                f"yaw_rate={v.r:+.2f}rad/s  "
                f"ay={v.ay_body():+.2f}m/s^2  "
                f"ax={v.ax_body():+.2f}m/s^2  "
                f"steer={car.last_cmd.steer:+.3f}rad"
            ),
            True, (200, 210, 220),
        )
        surf.blit(line, (self.x + pad, self.y + pad + big.get_height() + 4))

        abs_s = "ABS:ON " if car.abs_enabled else "ABS:OFF"
        stab_s = "ESC:ON " if car.stability_enabled else "ESC:OFF"
        fsm_letters = " ".join(f"{tag}={s.value}" for tag, s in zip(
            ("FL", "FR", "RL", "RR"), car.last_abs_states
        ))
        line2 = font.render(
            f"{abs_s}  {stab_s}  states: {fsm_letters}  surface: {car.last_surface}",
            True, (200, 230, 200),
        )
        surf.blit(line2, (self.x + pad, self.y + pad + big.get_height() + 4 + line.get_height() + 4))

        # Help line on the right
        help_text = [
            "SPACE: emergency brake",
            "A: toggle ABS   E: toggle ESC",
            "1-5: dry/wet/snow/ice/sand",
            "T: cycle tracks   R: reset",
        ]
        ry = self.y + pad
        for l in help_text:
            h = font_small.render(l, True, (160, 170, 180))
            surf.blit(h, (self.x + self.w - h.get_width() - pad, ry))
            ry += h.get_height() + 1

        if extra_lines:
            ry = self.y + self.h - pad - len(extra_lines) * 16
            for l in extra_lines:
                h = font_small.render(l, True, (200, 200, 140))
                surf.blit(h, (self.x + pad, ry))
                ry += 15
