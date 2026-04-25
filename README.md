# ABS Simulator

A Python simulator for an **Anti-lock Braking System (ABS)** on a 4-wheel car.

It models:

- 6-DOF chassis + per-wheel spin dynamics with a **Dugoff combined-slip tire
  model** (with post-peak friction fade so ABS actually helps).
- A per-wheel **ABS controller = finite-state machine (APPLY / HOLD / RELEASE)
  wrapped around a slip-target PID**, which is how real Bosch-style ABS works.
- A **chassis yaw-rate stability controller** (simplified ESC) that can override
  per-wheel brake pressures via differential braking when yaw error is large.
- A **brake actuator** with a first-order hydraulic lag + rate limit (that's
  where the "low-latency" non-functional requirement is a tunable parameter).
- **Tracks** made of straight + arc segments with per-segment surface friction,
  plus optional "surface patches" (e.g. an ice strip on a dry straight).
- **Drivers** that cruise along the centerline (pure-pursuit steering + PI
  speed control), brake for corners with a configurable delay, or slam random
  full-stop brake events.
- An interactive **pygame** application with a top-down track view, a 2x2 per-
  wheel quad panel, a HUD, and keyboard + slider controls.
- Post-run **matplotlib** reports generated from telemetry CSVs.

```
                       +--------------------+        +----------------+
  driver ------------->|  per-wheel ABS     |------> |    brake       |
  (throttle/brake/steer)|  FSM + slip PID   |        |    actuator    |----+
                       +--------------------+        +----------------+    |
                                ^                                          |
                                |                                          v
                    +-----------+--------+                         +-----------+
                    |  chassis stability |<------yaw rate----------|  vehicle  |
                    |   PID override     |                         | (physics) |
                    +--------------------+                         +-----------+
                                                                         ^
                                                                         | surface
                                                               +---------------+
                                                               |    track      |
                                                               +---------------+
```

## Requirements

- Python 3.9+
- `pygame-ce`, `pygame_gui`, `numpy`, `pyyaml`, `pandas`, `matplotlib`, `pytest`

Install everything with:

```bash
pip install -r requirements.txt
```

(`pygame-ce` is a drop-in replacement for `pygame` required by the newer
`pygame_gui` release — if you already have `pygame` installed from another
project, uninstall it first with `pip uninstall pygame` so the import resolves
to `pygame-ce`.)

## Running the interactive simulator

```bash
python scripts/run_interactive.py                 # default f1_like track
python scripts/run_interactive.py --track oval
python scripts/run_interactive.py --track curve_braking --cruise 35
python scripts/run_interactive.py --track straight --surface ice
python scripts/run_interactive.py --track split_mu_curves
```

### Keyboard controls

| Key     | Action                                                     |
|---------|------------------------------------------------------------|
| `SPACE` | Slam full brake on the ego car for 2 s                     |
| `1`..`5`| Override road surface (dry / wet / snow / ice / sand) for 4 s |
| `A`     | Toggle ABS on the ego car                                  |
| `E`     | Toggle chassis stability (ESC-like) on the ego car         |
| `T`     | Cycle to the next track preset                             |
| `R`     | Reset the current track                                    |
| `F`     | Toggle camera follow vs fit-track                          |
| `P`     | Cycle the ego car through the driver persona archetypes    |

### Sliders (right-hand panel)

- `ABS target slip` (`lambda_opt`) — where the mu-slip peak is targeted.
  Default 0.15, typical range 0.10-0.20.
- `ABS PID Kp`, `ABS PID Ki` — fine-modulation gains inside the APPLY state.
- `Actuator tau` — hydraulic lag, smaller = snappier brake response.
- `ESC PID Kp` — strength of the yaw-stability override.
- `Driver target speed` — sets the cruise driver's set-point.

## Headless demos (produce CSVs + plots)

```bash
# ABS on vs ABS off, straight-line brake from 20 m/s on snow.
python scripts/run_abs_compare.py --surface snow --v 20

# Three drivers entering the SAME corner at different brake timings (wet road).
python scripts/run_three_driver_demo.py --cruise 40 --corner 14 --surface wet --abs both

# Random road-surface patches + random full-stop brake events.
python scripts/run_random_surface_demo.py --seed 7

# Four driver-persona archetypes (Pro / Cautious / Novice / Aggressive)
# on the same track; produces per-persona telemetry + a comparison plot.
python scripts/run_persona_comparison.py --track f1_like
python scripts/run_persona_comparison.py --track split_mu_curves --only pro,cautious
```

Outputs go under `runs/` by default (configurable with `--out`). Each run
drops a CSV of all telemetry (per-wheel kappa, alpha, Fz, forces, omega, brake
command and actual pressure, ABS state letters, chassis accelerations, etc.)
and matplotlib comparison figures.

## Driver personas

`PersonaDriver` extends the cruise driver with two independent timing
offsets — `turn_lead_s` for steering and `brake_lead_s` for braking — so
the same track exercises materially different driving styles. Four named
archetypes ship in [`abs_sim/drivers/policies.py`](abs_sim/drivers/policies.py)
via the `PERSONAS` factory registry:

| Archetype  | `turn_lead_s` | `brake_lead_s` | `v_cruise` | `mu_assumed` | Character                                                   |
|------------|---------------|----------------|------------|--------------|-------------------------------------------------------------|
| Pro        | 0.0           | 0.0            | 18         | 0.85         | Clean, on-time; uses the track's actual grip accurately.    |
| Cautious   | +0.4 (early)  | +0.5 (early)   | 14         | 0.70         | Anticipates everything, slower cruise, lots of margin.      |
| Novice     | -0.3 (late)   | -0.4 (late)    | 18         | 0.90         | Reactive turn-in, brakes late, slightly overconfident.      |
| Aggressive | -0.2 (late)   | -0.6 (late)    | 22         | 0.95         | Fast cruise, trail-brakes deep, leaves no margin.           |

`turn_lead_s` shifts pure-pursuit lookahead by `turn_lead_s * speed`
meters (positive = looks further ahead = TURNS EARLY). `brake_lead_s`
adds `brake_lead_s * v_cruise` meters to the planner's `settle_buffer`
(positive = bigger pre-corner runway = BRAKES EARLY). The other two
traits — `v_cruise` and `mu_assumed` — round out each archetype's
character.

Exposed two ways:

- **Interactive:** Press `P` in the pygame app to cycle the ego car
  through Pro → Cautious → Novice → Aggressive → Pro. The HUD prints a
  one-line indicator showing the active persona and its trait values.
  The slider `v_cruise` is preserved across the cycle so only the
  timing/mu changes.
- **Offline:** `python scripts/run_persona_comparison.py` runs every
  persona on the same track, dumps per-persona CSVs, and renders a
  side-by-side comparison plot (speed vs. time, pedal vs. time, XY
  trajectory). Use `--only pro,aggressive` to pick a subset and
  `--track split_mu_curves` to compare on low-grip asphalt.

## Tests

```bash
python -m pytest
```

67 tests cover: tire-force curves and friction circle, wheel/chassis kinematics
and load transfer, RK4 integration sanity, ABS FSM transitions, ABS stopping
distance strictly less than no-ABS on low mu, ESC intervention on under- and
over-steer, track geometry (closed loops, arc endpoints, lateral offset sign),
YAML loading, driver policies, event scheduling, and end-to-end simulation
smoke tests.

## What's in each module

- [`abs_sim/physics/tire.py`](abs_sim/physics/tire.py) — Dugoff combined-slip
  model with surface-specific peak friction and a post-peak slide-friction
  fade (what makes ABS strictly better than locking up).
- [`abs_sim/physics/wheel.py`](abs_sim/physics/wheel.py) — wheel velocity-in-
  tire-frame transforms and SAE slip-ratio / slip-angle formulas.
- [`abs_sim/physics/chassis.py`](abs_sim/physics/chassis.py) — 6-DOF chassis
  parameters and quasi-static longitudinal + lateral load transfer.
- [`abs_sim/physics/vehicle.py`](abs_sim/physics/vehicle.py) — 10-state RK4
  integrator that glues chassis + 4 wheels + tire forces.
- [`abs_sim/control/brake_actuator.py`](abs_sim/control/brake_actuator.py) —
  rate-limited first-order hydraulic lag.
- [`abs_sim/control/wheel_abs.py`](abs_sim/control/wheel_abs.py) — per-wheel
  FSM (APPLY/HOLD/RELEASE) + slip-target PID. This is the ABS controller.
- [`abs_sim/control/stability.py`](abs_sim/control/stability.py) — chassis
  yaw-rate PID override (ESC-style differential braking).
- [`abs_sim/control/pid.py`](abs_sim/control/pid.py) — minimal PID with
  anti-windup integrator clamp.
- [`abs_sim/track/track.py`](abs_sim/track/track.py) — segment-based track
  with centerline sampling, closest-point lookup, and surface patches.
- [`abs_sim/track/presets.py`](abs_sim/track/presets.py) — straight, oval,
  figure-8, f1-like, curve-braking, random-surface-straight.
- [`abs_sim/track/loader.py`](abs_sim/track/loader.py) — YAML loader; see
  `abs_sim/config/tracks/*.yaml`.
- [`abs_sim/drivers/policies.py`](abs_sim/drivers/policies.py) — `Driver`
  base class plus `CruisePursuitDriver`, `CurveBrakeDelayDriver`,
  `RandomBrakeEventDriver`, and `PersonaDriver` (Pro / Cautious / Novice
  / Aggressive archetypes registered in `PERSONAS`).
- [`abs_sim/sim/simulation.py`](abs_sim/sim/simulation.py) — multi-car
  orchestrator with multi-rate scheduling (1 ms physics, 5 ms control, 60 Hz
  render) and telemetry logging.
- [`abs_sim/sim/events.py`](abs_sim/sim/events.py) — time-ordered event
  queue + canned event factories (`set_surface_override`, `force_brake`,
  `toggle_abs`, ...).
- [`abs_sim/sim/telemetry.py`](abs_sim/sim/telemetry.py) — in-memory or
  streaming CSV telemetry logger.
- [`abs_sim/viz/*`](abs_sim/viz/) — pygame app shell + top-down renderer +
  2x2 wheel panel + HUD + matplotlib reports.

## Algorithms that the code is based on

- **ABS control strategy**: FSM around slip-target PID. The FSM phases
  mirror real Bosch / NHTSA ABS pump cycles (apply pressure until slip peaks,
  dump pressure to let the wheel recover, hold, re-apply). The PID inside the
  APPLY phase targets the peak of the mu-slip curve (lambda_opt ≈ 0.15).
  References:
  - MathWorks, [Model an Anti-Lock Braking System](https://www.mathworks.com/help/simulink/slref/modeling-an-anti-lock-braking-system.html) (bang-bang ABS demo).
  - PathSim [ABS event-driven example](https://pathsim.readthedocs.io/en/v0.15.2/examples/abs_braking.html).
  - Automotive Experiences (2024), "Predictive Performance of ABS with PID Controller Optimized by GSA."
- **Tire model**: Dugoff combined-slip force with friction-ellipse saturation,
  parameterised by (Cx, Cy, mu, Fz). Augmented with a slide-friction fade
  (mu_slide < mu_peak) post-peak. See Dugoff, Fancher, Segel, 1970, and
  "A Modified Dugoff Tire Model for Combined-slip Forces" (Ding & Taheri, 2010).
- **Chassis**: planar (x, y, yaw + body velocities) with quasi-static load
  transfer, which is the standard control-oriented vehicle-dynamics
  simplification (no integrated suspension states).
- **Stability control**: linear-Ackermann reference yaw rate
  `r_des = V * delta / (L + K_us * V^2)` plus a PID that applies a differential
  brake bias on left vs right wheels — a simplified ESC.

## Known simplifications / non-goals

- No engine/drivetrain model beyond throttle-to-torque linear mapping, no
  transmission gears.
- Only aerodynamic drag (no downforce, wind, or yaw-sensitive aero).
- Suspension load transfer is quasi-static (no spring-damper states).
- Slip ratios can grow large during stand-still acceleration events (the
  denominator is floored at 0.5 m/s); tire forces are bounded correctly but
  the reported `kappa` values on accel-from-rest can exceed 1.
- Single ESC strategy (symmetric left/right bias). A full ESC would pick a
  specific wheel to brake depending on which failure mode (understeer /
  oversteer) it diagnoses.
