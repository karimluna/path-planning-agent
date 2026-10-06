import numpy as np
from models.base import KinematicBase

class SimpleCar(KinematicBase):
    n_state, n_control = 3, 2
    control_names = ("v", "ω")

    def __init__(self, L: float = 3.0) -> None:
        self.L = L

    def f(self, x, u):
        theta  = x[2]
        v, w   = u
        tan_w = np.tan(w)
        xdot   = np.array([v*np.cos(theta), v*np.sin(theta), (v/self.L)*tan_w])
        diag = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v [m/s]": v,
            "w [deg]": np.degrees(w),
            "R [m]": self.L / tan_w if abs(tan_w) > 1e-9 else float("inf"),
        }
        return xdot, diag