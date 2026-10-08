from __future__ import annotations

from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pydantic import Field
from pydantic.dataclasses import dataclass

from models.base import Controller
# my style!
plt.style.use("media/default_plt.mplstyle")

for _k in ("keymap.back", "keymap.forward", "keymap.home"):
    plt.rcParams[_k] = []


@dataclass(config={"arbitrary_types_allowed": True})
class AnimParams:
    dt: float = Field(default=0.05, description="Integration period [s]")
    window: float = Field(default=10.0, description="Seconds of signal history shown")
    view: float = Field(default=10.0, description="Half-width of the view [m]")
    head: float = Field(default=1.0, description="Heading line length [m]")
    follow: bool = Field(default=False, description="Camera follows the robot(s)")
    steps_per_frame: int = Field(default=1, description="Sim steps per drawn frame")
    trail: int = Field(default=2000, description="Max trail points drawn")
    plot_keys: tuple[str, str] = Field(
        default=("v [m/s]", "omega [rad/s]"),
        description="diag keys shown in the two side plots",
    )
    u_max: np.ndarray = Field(
        default_factory=lambda: np.array([2.0, 2.0]),
        description="Symmetric control limits, shape (n_control,)",
    )


class _Run:
    """One model's state, histories and artists."""

    def __init__(self, model, x0, u_max, color, keys):
        self.model, self.color, self.keys = model, color, keys
        self.name = type(model).__name__
        self.x0 = np.zeros(model.n_state) if x0 is None else np.asarray(x0, float)
        self.u_max = np.asarray(u_max, float)
        self.reset()

    def reset(self):
        self.x = self.x0.copy()
        self.traj = [self.x[:2].copy()]
        self.sig = {k: [] for k in self.keys}


class AnimTrajectory:
    def __init__(self, models, controller: Controller,
                 params: AnimParams | None = None, x0s=None, u_maxs=None) -> None:
        models = [models] if hasattr(models, "step") else list(models)
        if len({m.n_control for m in models}) != 1:
            raise ValueError("All compared models must have the same n_control")
        self.params = p = params or AnimParams()
        self.controller = controller
        x0s = x0s or [None] * len(models)
        u_maxs = u_maxs or [p.u_max] * len(models)
        if len(u_maxs) != len(models) or len(x0s) != len(models):
            raise ValueError("u_maxs and x0s need one entry per model.")
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.runs = [
            _Run(m, x0, um, colors[i % len(colors)], p.plot_keys)
            for i, (m, x0, um) in enumerate(zip(models, x0s, u_maxs))
        ]
        self._build_figure()
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.ts, self.us = [], []
        for r in self.runs:
            r.reset()
        self.controller.reset()
        for ax in (self.ax_a, self.ax_b):
            ax.set_ylim(-1, 1)

    def _build_figure(self) -> None:
        p = self.params
        self.fig = plt.figure(figsize=(11, 7))
        gs = GridSpec(4, 2, width_ratios=[2, 1], figure=self.fig)
        self.ax_xy = self.fig.add_subplot(gs[0:3, 0])
        self.ax_u = self.fig.add_subplot(gs[3, 0])
        self.ax_a = self.fig.add_subplot(gs[0:2, 1], sharex=self.ax_u)
        self.ax_b = self.fig.add_subplot(gs[2:4, 1], sharex=self.ax_u)
        self.ax_a.tick_params(labelbottom=False)

        for r in self.runs:
            (r.path,) = self.ax_xy.plot([], [], lw=1.5, color=r.color, label=r.name)
            (r.robot,) = self.ax_xy.plot([], [], "o", color=r.color)
            (r.heading,) = self.ax_xy.plot([], [], "-", lw=2, color=r.color)
            r.lines = [ax.plot([], [], color=r.color, label=r.name)[0]
                       for ax in (self.ax_a, self.ax_b)]
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlim(-p.view, p.view)
        self.ax_xy.set_ylim(-p.view, p.view)
        self.ax_xy.set_xlabel("x [m]")
        self.ax_xy.set_ylabel("y [m]")
        self.ax_xy.grid(True)
        self.ax_xy.legend(loc="upper left")

        first = self.runs[0].model
        names = (getattr(first, "control_names", None) if len(self.runs) == 1 else None) \
            or [f"u{i}" for i in range(first.n_control)]
        self.u_lines = [self.ax_u.plot([], [], label=n)[0] for n in names]
        lim = max(float(np.max(r.u_max)) for r in self.runs) * 1.1
        self.ax_u.set_ylim(-lim, lim)
        self.ax_u.set_xlim(0, p.window)
        self.ax_u.set_xlabel("t [s]")
        self.ax_u.set_ylabel("u")
        self.ax_u.legend(loc="upper left")
        self.ax_u.grid(True)

        for ax, k in zip((self.ax_a, self.ax_b), p.plot_keys):
            ax.set_ylabel(k)
            ax.grid(True)
        self.ax_b.set_xlabel("t [s]")
        if len(self.runs) > 1:
            self.ax_a.legend(loc="upper left")

        self.fig.suptitle("hold ↑/↓ u0   hold ←/→ u1   r: reset", fontsize=9)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

    def _on_key(self, event) -> None:
        if event.key == "r":
            self.reset()
        elif hasattr(self.controller, "on_key"):
            self.controller.on_key(event)

    def _on_key_release(self, event) -> None:
        if hasattr(self.controller, "on_key_release"):
            self.controller.on_key_release(event)

    @staticmethod
    def _fit_y(ax, series) -> None:
        """Grow-only y-limits: cheap, no per-frame autoscale."""
        lo = min(min(s) for s in series)
        hi = max(max(s) for s in series)
        pad = 0.1 * max(hi - lo, 1e-3)
        cur_lo, cur_hi = ax.get_ylim()
        if lo - pad < cur_lo or hi + pad > cur_hi:
            ax.set_ylim(min(cur_lo, lo - pad), max(cur_hi, hi + pad))

    def update(self, _frame):
        p = self.params
        for _ in range(p.steps_per_frame):
            self.ts.append(self.t)               # signals are sampled at the start of the step
            for i, r in enumerate(self.runs):
                u = np.clip(self.controller(self.t, r.x), -r.u_max, r.u_max)
                r.x, diag = r.model.step(r.x, u, p.dt)
                r.traj.append(r.x[:2].copy())
                for k in p.plot_keys:
                    r.sig[k].append(diag[k])
                if i == 0:
                    self.us.append(u.copy())
            self.t += p.dt

        for r in self.runs:
            x = r.x
            tr = np.asarray(r.traj[-p.trail:])
            r.path.set_data(tr[:, 0], tr[:, 1])
            r.robot.set_data([x[0]], [x[1]])
            r.heading.set_data([x[0], x[0] + p.head * np.cos(x[2])],
                               [x[1], x[1] + p.head * np.sin(x[2])])
        if p.follow:
            cx, cy = np.mean([r.x[:2] for r in self.runs], axis=0)
            self.ax_xy.set_xlim(cx - p.view, cx + p.view)
            self.ax_xy.set_ylim(cy - p.view, cy + p.view)

        n = int(p.window / p.dt)
        tt = np.asarray(self.ts[-n:])
        uu = np.asarray(self.us[-n:])
        for i, ln in enumerate(self.u_lines):
            ln.set_data(tt, uu[:, i])
        for r in self.runs:
            for ln, k in zip(r.lines, p.plot_keys):
                ln.set_data(tt, r.sig[k][-n:])
        for ax, k in zip((self.ax_a, self.ax_b), p.plot_keys):
            self._fit_y(ax, [r.sig[k][-n:] for r in self.runs])
        self.ax_u.set_xlim(max(0.0, self.t - p.window), max(p.window, self.t))
        return ()

    def show(self) -> None:
        self.anim = FuncAnimation(
            self.fig, self.update,
            interval=int(self.params.dt * 1000),
            blit=False, cache_frame_data=False,
        )
        plt.show()