"""Simulator CLI: send a kinematic model to a point, make it follow a circle, or drive it by hand.

Examples
  python main.py unicycle                          # go to the default target of that model
  python main.py car --target 20 10 --kp 1.5       # SimpleCar to (20, 10)
  python main.py ddr --circle 0 5 5 0.15           # follow a circle: centre x y, radius, rad/s
  python main.py smooth-ddr --kp 1.5 --kd 0.3      # PD on the tracked point
  python main.py skidding-car --target 100 50
  python main.py unicycle --start -5 3 --heading 90    # start at (-5, 3) facing +y
  python main.py unicycle --keyboard               # open-loop keys instead of a tracker
  python main.py --list                            # models and their defaults
  python main.py ddr --print-config                # show the resolved settings, no window
"""
from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass

import numpy as np

from models.any.car import ContinuousSteeringCar, SimpleCar, SkiddingCar, SteeringAccelCar
from models.any.ddr import DifferentialDriveRobot, SmoothDDR
from models.any.unicycle import BetterUnicycle, EvenBetterUnicycle, Unicycle
from models.controllers.keyboard import KeyboardController
from models.controllers.pid import PID
from models.controllers.tracking import PointTracker, circle, static_point
from utils.plots import AnimParams, AnimTrajectory


@dataclass(frozen=True)
class Spec:
    """Defaults for one model. u_max bounds the model's own input, so its meaning differs per model."""
    cls: type
    u_max: tuple
    amp: tuple                       # keyboard amplitude
    d: float                         # offset of the tracked point [m]
    view: float                      # half-width of the view [m]
    target: tuple = (8.0, 5.0)
    aliases: tuple = ()


REGISTRY: dict[str, Spec] = {
    "unicycle":                Spec(Unicycle,               (3, 2),      (2, 1),     0.3, 10, aliases=("uni",)),
    "better-unicycle":         Spec(BetterUnicycle,         (5, 2),      (2, 1),     0.3, 10),
    "even-better-unicycle":    Spec(EvenBetterUnicycle,     (5, 5),      (2, 1),     0.3, 10, aliases=("ebu",)),
    "ddr":                     Spec(DifferentialDriveRobot, (30, 30),    (10, 10),   0.3, 5),
    "smooth-ddr":              Spec(SmoothDDR,              (120, 120),   (10, 10),   0.3, 5),
    "simple-car":              Spec(SimpleCar,              (3, 0.6),    (2, 0.4),   0.6, 20, (20.0, 10.0), aliases=("car",)),
    "continuous-steering-car": Spec(ContinuousSteeringCar,  (5, 1.0),    (2, 0.4),   0.6, 20, (20.0, 10.0)),
    "steering-accel-car":      Spec(SteeringAccelCar,       (5, 10),     (2, 0.4),   0.6, 20, (20.0, 10.0)),
    "skidding-car":            Spec(SkiddingCar,            (0.15,),     (0.05,),    3.0, 150, (100.0, 50.0),
                                    aliases=("skid",)),
}
ALIASES = {a: name for name, s in REGISTRY.items() for a in (name, *s.aliases)}


def model_name(text: str) -> str:
    key = text.strip().lower().replace("_", "-").replace(" ", "-")
    if key in ALIASES:
        return ALIASES[key]
    hint = difflib.get_close_matches(key, ALIASES, n=1)
    msg = f"unknown model '{text}'" + (f" - did you mean '{hint[0]}'?" if hint else "")
    raise argparse.ArgumentTypeError(msg + "  (see --list)")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py", description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", type=model_name, metavar="MODEL",
                   help="one of: " + ", ".join(REGISTRY) + "  (short: car, uni, ebu, skid)")
    p.add_argument("--list", action="store_true", help="list the models and their defaults, then exit")
    p.add_argument("--print-config", action="store_true", help="show the resolved settings and exit (no window)")

    g = p.add_argument_group("what to track")
    ex = g.add_mutually_exclusive_group()
    ex.add_argument("--target", nargs=2, type=float, metavar=("X", "Y"),
                    help="fixed point [m] (default depends on the model)")
    ex.add_argument("--circle", nargs=4, type=float, metavar=("CX", "CY", "R", "W"),
                    help="point moving on a circle: centre [m], radius [m], angular speed [rad/s]")
    g.add_argument("--phase", type=float, default=0.0, metavar="DEG",
                   help="where on the circle the point starts, from +x [deg] (default 0)")
    g.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"), help="initial position [m] (default 0 0)")
    g.add_argument("--heading", type=float, default=0.0, metavar="DEG", help="initial heading [deg] (default 0 = +x)")

    g = p.add_argument_group("controller (PID on a point d ahead of the axle; feedforward is the target's velocity)")
    g.add_argument("--kp", type=float, default=1.0, help="proportional gain [1/s] (default 1.0)")
    g.add_argument("--kd", type=float, default=0.0,
                   help="derivative gain, dimensionless; keep <= 0.5 at dt=0.05 (default 0)")
    g.add_argument("--ki", type=float, default=0.0, help="integral gain (implemented, not yet tested) (default 0)")
    g.add_argument("--i-max", type=float, default=np.inf, metavar="M", help="clamp on the integral term (anti-windup)")
    g.add_argument("--d", type=float, default=None, help="offset of the tracked point [m] (default per model)")
    g.add_argument("--v-max", type=float, default=None, metavar="V", help="cap on the desired point speed [m/s]")
    g.add_argument("--ks", type=float, default=None, help="speed-loop gain of models with a speed state (default 4)")
    g.add_argument("--kw", type=float, default=None, help="yaw/wheel-rate loop gain (default 8)")
    g.add_argument("--kphi", type=float, default=None, help="steering-angle loop gain (default 4)")

    g = p.add_argument_group("manual driving")
    g.add_argument("--keyboard", action="store_true", help="open-loop keys (up/down: u0, left/right: u1) instead of a tracker")
    g.add_argument("--amp", nargs="+", type=float, metavar="A", help="keyboard amplitude per input (default per model)")

    g = p.add_argument_group("limits")
    g.add_argument("--u-max", nargs="+", type=float, metavar="U",
                   help="bound on each input of the model, in its own units (default per model; see --list)")

    g = p.add_argument_group("viewer")
    g.add_argument("--dt", type=float, default=0.05, help="integration step [s] (default 0.05)")
    g.add_argument("--view", type=float, default=None, help="half-width of the view [m] (default per model)")
    g.add_argument("--follow", action=argparse.BooleanOptionalAction, default=True, help="camera follows the robot")
    g.add_argument("--window", type=float, default=10.0, help="seconds of signal history shown (default 10)")
    g.add_argument("--steps-per-frame", type=int, default=1, metavar="N", help="simulation steps per drawn frame")
    g.add_argument("--trail", type=int, default=2000, help="maximum trail points drawn")
    return p


def list_models() -> None:
    print(f"{'model':26s}{'short':8s}{'states':>7s}{'inputs':>7s}  {'u_max':16s}{'d [m]':>6s}{'view':>6s}  default target")
    for name, s in REGISTRY.items():
        print(f"{name:26s}{','.join(s.aliases):8s}{s.cls.n_state:7d}{s.cls.n_control:7d}  "
              f"{str(list(s.u_max)):16s}{s.d:6.1f}{s.view:6.0f}  {s.target}")


def parse(argv=None) -> argparse.Namespace:
    p = make_parser()
    a = p.parse_args(argv)
    if a.list:
        list_models()
        raise SystemExit(0)
    if a.model is None:
        p.error("MODEL is required (try --list)")
    spec = REGISTRY[a.model]
    n = spec.cls.n_control
    for flag, vals, want in (("--u-max", a.u_max, n), ("--amp", a.amp, n)):
        if vals is not None and len(vals) != want:
            p.error(f"{a.model} has {want} input(s); {flag} needs {want} value(s), got {len(vals)}")
    if a.u_max is not None and min(a.u_max) <= 0:
        p.error("--u-max values must be positive")
    if a.d is not None and a.d <= 0:
        p.error("--d must be positive")
    if a.dt <= 0:
        p.error("--dt must be positive")
    if a.kd > 0.5:
        print(f"warning: kd={a.kd} > 0.5 can make the sampled derivative unstable at dt=0.05 "
              "(it diverged at 0.9 in my tests)", file=sys.stderr)
    return a


def build(a: argparse.Namespace):
    spec = REGISTRY[a.model]
    model = spec.cls()
    u_max = np.array(a.u_max if a.u_max is not None else spec.u_max, float)

    x0 = None
    if a.start is not None or a.heading != 0.0:
        x0 = np.zeros(model.n_state)
        x0[:2] = a.start if a.start is not None else (0.0, 0.0)
        x0[2] = np.deg2rad(a.heading)

    ref = None
    if a.keyboard:
        ctrl = KeyboardController(model.n_control, amp=np.array(a.amp if a.amp is not None else spec.amp, float))
    else:
        ref = (circle(a.circle[:2], a.circle[2], a.circle[3], phase=np.deg2rad(a.phase)) if a.circle
               else static_point(a.target if a.target else spec.target))
        gains = {k: v for k, v in (("ks", a.ks), ("kw", a.kw), ("kphi", a.kphi)) if v is not None}
        ctrl = PointTracker(model, ref, PID(kp=a.kp, ki=a.ki, kd=a.kd, i_max=a.i_max),
                            d=a.d if a.d is not None else spec.d,
                            v_max=a.v_max if a.v_max is not None else np.inf, **gains)

    params = AnimParams(dt=a.dt, window=a.window, view=a.view if a.view is not None else spec.view,
                        follow=a.follow, steps_per_frame=a.steps_per_frame, trail=a.trail)
    return model, ctrl, params, u_max, x0, ref


def describe(a: argparse.Namespace, model, params, u_max, x0) -> str:
    spec = REGISTRY[a.model]
    what = ("keyboard (open loop)" if a.keyboard else
            f"circle centre {a.circle[:2]}, radius {a.circle[2]}, {a.circle[3]} rad/s, phase {a.phase} deg" if a.circle else
            f"fixed point {tuple(a.target) if a.target else spec.target}")
    lines = [f"model        {a.model}  ({type(model).__name__}, {model.n_state} states, {model.n_control} input(s))",
             f"task         {what}",
             f"start        {'zeros' if x0 is None else np.round(x0[:3], 3).tolist()}"]
    if not a.keyboard:
        lines += [f"controller   kp={a.kp} ki={a.ki} kd={a.kd}  d={a.d if a.d is not None else spec.d} m  "
                  f"v_max={'off' if a.v_max is None else a.v_max}"]
    lines += [f"u_max        {u_max.tolist()}", f"viewer       dt={params.dt} view={params.view} follow={params.follow}"]
    return "\n".join(lines)


def main(argv=None) -> None:
    a = parse(argv)
    model, ctrl, params, u_max, x0, ref = build(a)
    print(describe(a, model, params, u_max, x0))
    if a.print_config:
        return
    AnimTrajectory(model, ctrl, params, x0s=[x0] if x0 is not None else None, u_maxs=[u_max], ref=ref).show()


if __name__ == "__main__":
    main()