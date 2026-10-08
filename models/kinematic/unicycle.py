import numpy as np
from models.base import KinematicBase

class Unicycle(KinematicBase):
    n_state, n_control = 3, 2
    control_names = ("v", "ω")

    def f(self, x, u):
        theta    = x[2]
        v, w = u
        xdot     = np.array([v*np.cos(theta), v*np.sin(theta), w])
        diag     = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v [m/s]": v, 
            "w [rad/s]": w,
        }
        return xdot, diag


class BetterUnicycle(KinematicBase):
    n_state, n_control = 4, 2
    control_names = ("u_w", "u_a")

    def f(self, x, u):
        s        = x[3]
        theta    = x[2]
        u_w, u_a = u
        xdot     = np.array([s*np.cos(theta), s*np.sin(theta), u_w, u_a])
        diag     = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "u_w [m/s]": u_w, 
            "u_a [rad/s]": u_a,
        }
        return xdot, diag
    

class EvenBetterUnicycle(KinematicBase):
    n_state, n_control = 5, 2
    control_names = ("u_w", "u_a")

    def f(self, x, u):
        omega    = x[4]
        s        = x[3]
        theta    = x[2]
        u_w, u_a = u
        xdot     = np.array([s*np.cos(theta), s*np.sin(theta), omega, u_w, u_a])
        diag     = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "u_w [m/s]": u_w, 
            "u_a [rad/s]": u_a,
        }
        return xdot, diag
