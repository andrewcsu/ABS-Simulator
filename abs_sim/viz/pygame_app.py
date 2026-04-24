"""Main interactive pygame app: top-down + quad wheel panel + HUD + controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math
import sys

import pygame
import pygame_gui

from abs_sim.drivers.policies import CruisePursuitDriver, Driver
from abs_sim.physics.tire import SURFACES
from abs_sim.sim.events import force_brake, set_surface_override
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.track.presets import PRESETS, oval
from abs_sim.track.track import Track
from abs_sim.viz.camera import Camera
from abs_sim.viz.hud import HUDRenderer
from abs_sim.viz.top_down import TireMarkBuffer, TrackRenderer
from abs_sim.viz.wheel_panel import WheelPanelRenderer


WIN_W, WIN_H = 1500, 900

TOPDOWN_RECT = (20, 20, 900, 620)
WHEEL_RECT = (940, 20, 540, 420)
HUD_RECT = (20, 660, 900, 220)
CONTROL_RECT = (940, 460, 540, 420)


DEFAULT_TRACK_ORDER = ["oval", "figure_8", "f1_like", "curve_braking",
                       "straight", "random_surface_straight"]


@dataclass
class AppOptions:
    track_preset: str = "f1_like"
    v_cruise: float = 30.0
    follow_camera: bool = True
    telemetry_path: Optional[str] = None
    show_multi_car: bool = False
    multi_cars: List[Tuple[str, Driver, Tuple[int, int, int]]] = field(default_factory=list)
    initial_surface_override: Optional[str] = None


class PygameApp:
    def __init__(self, options: AppOptions) -> None:
        pygame.init()
        pygame.display.set_caption("ABS Simulator")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.options = options

        self._track_order = DEFAULT_TRACK_ORDER[:]
        if options.track_preset not in self._track_order:
            self._track_order.insert(0, options.track_preset)
        self._track_idx = self._track_order.index(options.track_preset)

        self.ui = pygame_gui.UIManager((WIN_W, WIN_H))

        self.sim: Optional[Simulation] = None
        self.track_renderer: Optional[TrackRenderer] = None
        self.tire_marks = TireMarkBuffer()
        self.wheel_panel = WheelPanelRenderer(WHEEL_RECT)
        self.hud = HUDRenderer(HUD_RECT)
        self.topdown_cam = Camera(
            w=TOPDOWN_RECT[2], h=TOPDOWN_RECT[3],
            viewport_x=TOPDOWN_RECT[0], viewport_y=TOPDOWN_RECT[1],
        )
        self._extra_lines: List[str] = []

        self._build_ui()
        self._load_track(self._track_order[self._track_idx])

    # --------------------------------------------------------------- #
    # Track / sim (re)build
    # --------------------------------------------------------------- #
    def _load_track(self, name: str) -> None:
        track = PRESETS[name]() if name in PRESETS else oval()
        self.track_renderer = TrackRenderer(track)
        self.tire_marks = TireMarkBuffer()

        cars = self._build_cars()
        for c in cars:
            c.vehicle.set_pose(0.0, 0.0, 0.0)
            c.vehicle.set_speed(self.options.v_cruise)
        self.sim = Simulation(track=track, cars=cars)
        if self.options.follow_camera:
            self.topdown_cam.fit(track.centerline_points(), padding=40)
        else:
            self.topdown_cam.fit(track.centerline_points(), padding=40)

        if self.options.initial_surface_override:
            for c in cars:
                c.surface_override = self.options.initial_surface_override

    def _build_cars(self) -> List[Car]:
        if self.options.show_multi_car and self.options.multi_cars:
            out: List[Car] = []
            for name, drv, col in self.options.multi_cars:
                c = Car.make_default(name=name, driver=drv, color=col)
                out.append(c)
            return out
        car = Car.make_default(
            name="ego",
            driver=CruisePursuitDriver(v_cruise=self.options.v_cruise),
            color=(80, 200, 255),
        )
        return [car]

    # --------------------------------------------------------------- #
    # UI widgets
    # --------------------------------------------------------------- #
    def _build_ui(self) -> None:
        cx, cy, cw, ch = CONTROL_RECT
        self._sliders: Dict[str, pygame_gui.elements.UIHorizontalSlider] = {}
        self._labels: Dict[str, pygame_gui.elements.UILabel] = {}

        title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(cx, cy - 20, cw, 22),
            text="Controls",
            manager=self.ui,
        )

        slider_configs = [
            ("lambda_opt", "ABS target slip", 0.02, 0.35, 0.15, 0.01),
            ("pid_kp", "ABS PID Kp", 0.0, 10.0, 3.0, 0.1),
            ("pid_ki", "ABS PID Ki", 0.0, 200.0, 40.0, 1.0),
            # Slider default MUST match BrakeActuator.tau (0.008 s). Previously
            # this label read 0.03 while the actuator was initialized to 0.008,
            # so any stray click on the slider silently slowed the brake
            # modulator by ~4x and kept ABS from dumping pressure -> rear
            # lockup. Upper bound kept at 0.15 for exploration.
            ("actuator_tau", "Actuator tau (s)", 0.005, 0.15, 0.008, 0.001),
            ("stab_kp", "ESC PID Kp", 0.0, 3.0, 0.5, 0.05),
            ("driver_v", "Driver target speed (m/s)", 5.0, 60.0, 30.0, 0.5),
        ]
        y = cy + 20
        for key, label, lo, hi, default, step in slider_configs:
            self._labels[key] = pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(cx, y, cw, 18),
                text=f"{label}: {default:.3g}",
                manager=self.ui,
            )
            self._sliders[key] = pygame_gui.elements.UIHorizontalSlider(
                relative_rect=pygame.Rect(cx, y + 18, cw, 20),
                start_value=default,
                value_range=(lo, hi),
                manager=self.ui,
            )
            y += 46

        self._btn_abs = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(cx, y, cw // 2 - 4, 30),
            text="Toggle ABS",
            manager=self.ui,
        )
        self._btn_esc = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(cx + cw // 2 + 4, y, cw // 2 - 4, 30),
            text="Toggle ESC",
            manager=self.ui,
        )
        y += 36
        self._btn_reset = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(cx, y, cw, 30),
            text="Reset simulation",
            manager=self.ui,
        )

    # --------------------------------------------------------------- #
    # Input handling
    # --------------------------------------------------------------- #
    def _cycle_track(self, delta: int = 1) -> None:
        self._track_idx = (self._track_idx + delta) % len(self._track_order)
        self._load_track(self._track_order[self._track_idx])
        self._extra_lines = [f"Track: {self._track_order[self._track_idx]}"]

    def _handle_key(self, key: int) -> None:
        assert self.sim is not None
        car = self.sim.cars[0]
        if key == pygame.K_SPACE:
            self.sim.inject(force_brake(1.0, duration=2.0, car_index=0))
            self._extra_lines = ["[manual] emergency brake"]
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            order = ["dry", "wet", "snow", "ice", "sand"]
            idx = key - pygame.K_1
            s = order[idx]
            self.sim.inject(set_surface_override(s, duration=4.0, car_index=0))
            self._extra_lines = [f"[surface] override -> {s}"]
        elif key == pygame.K_a:
            car.abs_enabled = not car.abs_enabled
            self._extra_lines = [f"ABS -> {'ON' if car.abs_enabled else 'OFF'}"]
        elif key == pygame.K_e:
            car.stability_enabled = not car.stability_enabled
            self._extra_lines = [f"ESC -> {'ON' if car.stability_enabled else 'OFF'}"]
        elif key == pygame.K_t:
            self._cycle_track(+1)
        elif key == pygame.K_r:
            self._load_track(self._track_order[self._track_idx])
            self._extra_lines = ["reset"]
        elif key == pygame.K_f:
            self.options.follow_camera = not self.options.follow_camera
            if not self.options.follow_camera and self.track_renderer is not None:
                self.topdown_cam.fit(self.sim.track.centerline_points(), padding=40)

    def _handle_ui_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            for key, slider in self._sliders.items():
                if event.ui_element is slider:
                    v = slider.get_current_value()
                    self._apply_slider(key, v)
                    self._labels[key].set_text(
                        f"{self._labels[key].text.split(':')[0]}: {v:.3g}"
                    )
        elif event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self._btn_abs:
                self.sim.cars[0].abs_enabled = not self.sim.cars[0].abs_enabled
            elif event.ui_element is self._btn_esc:
                c = self.sim.cars[0]
                c.stability_enabled = not c.stability_enabled
            elif event.ui_element is self._btn_reset:
                self._load_track(self._track_order[self._track_idx])

    def _apply_slider(self, key: str, v: float) -> None:
        assert self.sim is not None
        for car in self.sim.cars:
            if key == "lambda_opt":
                for a in car.abs_controllers:
                    a.lambda_opt = v
            elif key == "pid_kp":
                for a in car.abs_controllers:
                    a._pid.kp = v
            elif key == "pid_ki":
                for a in car.abs_controllers:
                    a._pid.ki = v
            elif key == "actuator_tau":
                for b in car.actuators:
                    b.tau = v
            elif key == "stab_kp":
                car.stability._pid.kp = v
            elif key == "driver_v":
                drv = car.driver
                if hasattr(drv, "v_cruise"):
                    drv.v_cruise = v

    # --------------------------------------------------------------- #
    # Frame
    # --------------------------------------------------------------- #
    def run(self, max_frames: Optional[int] = None,
            on_frame: Optional[Callable[[int, "PygameApp"], None]] = None) -> None:
        running = True
        frame = 0
        try:
            while running:
                dt_real = self.clock.tick(60) / 1000.0

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        self._handle_key(event.key)
                    else:
                        self._handle_ui_event(event)
                    self.ui.process_events(event)

                self.ui.update(dt_real)

                self._advance_sim(dt_real)
                self._draw()
                pygame.display.flip()

                if on_frame is not None:
                    on_frame(frame, self)

                frame += 1
                if max_frames is not None and frame >= max_frames:
                    running = False
        finally:
            pygame.quit()

    def _advance_sim(self, dt_real: float) -> None:
        assert self.sim is not None
        # Cap to avoid spiraling on slow frames
        target_sim_dt = min(dt_real, 1.0 / 20.0)
        n_steps = max(1, int(target_sim_dt / self.sim.dt_phys))
        for _ in range(n_steps):
            self.sim.step()

        # Tire marks
        for car in self.sim.cars:
            self.tire_marks.maybe_add_from_car(car)

        if self.options.follow_camera and self.sim.cars:
            self.topdown_cam.follow(self.sim.cars[0].vehicle.x, self.sim.cars[0].vehicle.y)

    def _draw(self) -> None:
        assert self.sim is not None and self.track_renderer is not None
        self.screen.fill((12, 14, 18))

        self.track_renderer.draw_background(self.screen, self.topdown_cam)
        self.track_renderer.draw_track(self.screen, self.topdown_cam)
        for car in self.sim.cars:
            self.track_renderer.draw_car(
                self.screen, self.topdown_cam, car, self.tire_marks.points,
            )

        ego = self.sim.cars[0]
        self.wheel_panel.draw(self.screen, ego, dt_real=1.0 / 60.0)
        self.hud.draw(self.screen, ego, sim_time=self.sim.time,
                      extra_lines=self._extra_lines)

        # Controls background panel
        cx, cy, cw, ch = CONTROL_RECT
        pygame.draw.rect(self.screen, (20, 22, 26),
                         pygame.Rect(cx - 10, cy - 35, cw + 20, ch + 45))
        pygame.draw.rect(self.screen, (60, 62, 68),
                         pygame.Rect(cx - 10, cy - 35, cw + 20, ch + 45), 1)

        self.ui.draw_ui(self.screen)


def main() -> None:
    opts = AppOptions(track_preset="f1_like", v_cruise=30.0, follow_camera=True)
    PygameApp(opts).run()


if __name__ == "__main__":
    main()
