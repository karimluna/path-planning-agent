from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from pydantic import Field
from pydantic.dataclasses import dataclass

# my style!
plt.style.use("media/default_plt.mplstyle")

# free arrow keys / 'r' from matplotlib's default shortcuts
for _k in ("keymap.back", "keymap.forward", "keymap.home"):
    plt.rcParams[_k] = []


@dataclass(config={"arbitrary_types_allowed": True})
class AnimParams:
    dt: float = Field(default=0.05, description="Integration period [s]")
    window: float = Field(default=10.0, description="Seconds of signal history shown")
    view: float = Field(default=10.0, description="Half-width of the view [m]")
    head: float = Field(default=1.0, description="Heading line length [m]")
    follow: bool = Field(default=True, description="Camera follows the robot")
    steps_per_frame: int = Field(default=1, description="Sim steps per drawn frame")
    u_max: np.ndarray = Field(
        default_factory=lambda: np.array([2.0, 2.0]),
        description="Symmetric control limits, shape (n_control,)",
    )



class AnimTrajectory:
    def __init__(self, model, controller: Controller, params: AnimParams | None = None,
                 x0: np.ndarray | None = None) -> None:
        self.model = model
        self.controller = controller
        self.params = params or AnimParams()
        self.x0 = np.zeros(model.n_state) if x0 is None else np.asarray(x0, float)
        self._build_figure()
        self.reset()

    # state
    def reset(self) -> None:
        self.x = self.x0.copy()
        self.t = 0.0
        self.ts, self.us, self.traj = [], [], [self.x[:2].copy()]
        self.controller.reset()

    def _build_figure(self) -> None:
        p = self.params
        self.fig = plt.figure(figsize=(7, 7))
        gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1], figure=self.fig)
        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_u = self.fig.add_subplot(gs[1, 0])
        self.ax_d = self.fig.add_subplot(gs[:, 1])
        self.ax_d.axis("off")

        (self.path,) = self.ax_xy.plot([], [], lw=1.5)
        (self.robot,) = self.ax_xy.plot([], [], "ro")
        (self.heading,) = self.ax_xy.plot([], [], "r-", lw=2)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_xlim(-p.view, p.view)
        self.ax_xy.set_ylim(-p.view, p.view)
        self.ax_xy.set_xlabel("x [m]")
        self.ax_xy.set_ylabel("y [m]")
        self.ax_xy.grid(True)

        names = getattr(self.model, "control_names",
                        [f"u{i}" for i in range(self.model.n_control)])
        self.lines = [self.ax_u.plot([], [], label=n)[0] for n in names]
        lim = float(np.max(p.u_max)) * 1.1
        self.ax_u.set_ylim(-lim, lim)
        self.ax_u.set_xlim(0, p.window)
        self.ax_u.set_xlabel("t [s]")
        self.ax_u.set_ylabel("u")
        self.ax_u.legend(loc="upper left")
        self.ax_u.grid(True)

        self.diag_txt = self.ax_d.text(0.0, 1.0, "", va="top",
                                       family="monospace", fontsize=10)
        self.fig.suptitle("↑/↓ u0   ←/→ u1   space: zero   r: reset", fontsize=9)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event) -> None:
        if event.key == "r":
            self.reset()
        elif hasattr(self.controller, "on_key"):
            self.controller.on_key(event)

    def update(self, _frame):
        p = self.params
        for _ in range(p.steps_per_frame):
            u = np.clip(self.controller(self.t, self.x), -p.u_max, p.u_max)
            self.x, diag = self.model.step(self.x, u, p.dt)
            self.t += p.dt
            self.ts.append(self.t)
            self.us.append(np.array(u, copy=True))
            self.traj.append(self.x[:2].copy())

        x = self.x
        tr = np.asarray(self.traj)
        self.path.set_data(tr[:, 0], tr[:, 1])
        self.robot.set_data([x[0]], [x[1]])
        self.heading.set_data([x[0], x[0] + p.head * np.cos(x[2])],
                              [x[1], x[1] + p.head * np.sin(x[2])])
        if p.follow:
            self.ax_xy.set_xlim(x[0] - p.view, x[0] + p.view)
            self.ax_xy.set_ylim(x[1] - p.view, x[1] + p.view)

        n = int(p.window / p.dt)
        tt, uu = np.asarray(self.ts[-n:]), np.asarray(self.us[-n:])
        for i, ln in enumerate(self.lines):
            ln.set_data(tt, uu[:, i])
        self.ax_u.set_xlim(max(0.0, self.t - p.window), max(p.window, self.t))

        self.diag_txt.set_text(
            "Diagnostics\n" + "-" * 24 + "\n"
            + "\n".join(f"{k:<14}{v:>9.3f}" for k, v in diag.items())
        )
        return ()

    def show(self) -> None:
        self.anim = FuncAnimation(  # keep a reference or it gets garbage-collected
            self.fig, self.update,
            interval=int(self.params.dt * 1000),
            blit=False, cache_frame_data=False,
        )
        plt.show()