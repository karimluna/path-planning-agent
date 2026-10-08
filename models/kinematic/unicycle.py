import numpy as np

from models.base import KinematicBase


def _wrap_deg(theta):
    '''Return angle in degrees rather than radians.'''
    return np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))

class Unicycle(KinematicBase):
    n_state, n_control = 3, 2            # x = [x, y, θ],  u = [v, ω]
    control_names = ("v", "ω")

    def f(self, x, u):
        theta = x[2]
        v, w = u
        xdot = np.array([v*np.cos(theta), v*np.sin(theta), w])
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1], 
            "theta [deg]": _wrap_deg(theta),
            "v [m/s]": v, 
            "omega [rad/s]": w,
        }
        return xdot, diag


class BetterUnicycle(KinematicBase):
    n_state, n_control = 4, 2            # x = [x, y, θ, s],  u = [a, ω]
    control_names = ("a", "ω")
    S_MAX = 3.0

    def f(self, x, u):
        theta, s = x[2], x[3]
        a, w = u
        xdot = np.array([s*np.cos(theta), s*np.sin(theta), w, a - self.C_S*s])
        diag = {
            "x [m]": x[0], "y [m]": x[1], "theta [deg]": _wrap_deg(theta),
            "v [m/s]": s, "omega [rad/s]": w, "a [m/s²]": a,
        }
        return xdot, diag

    def step(self, x, u, dt):
        x_next, diag = super().step(x, u, dt)
        x_next[3] = np.clip(x_next[3], -self.S_MAX, self.S_MAX)
        return x_next, diag


class EvenBetterUnicycle(KinematicBase):
    n_state, n_control = 5, 2            # x = [x, y, θ, s, ω],  u = [a, α]
    control_names = ("a", "α")
    S_MAX, W_MAX = 3.0, 2.0

    def f(self, x, u):
        theta, s, w = x[2], x[3], x[4]
        a, alpha = u
        xdot = np.array([s*np.cos(theta), s*np.sin(theta), w,
                         a, alpha])
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1], 
            "theta [deg]": _wrap_deg(theta),
            "v [m/s]": s, 
            "omega [rad/s]": w,
            "a [m/s²]": a, 
            "alpha [rad/s²]": alpha,
        }
        return xdot, diag

    def step(self, x, u, dt):
        x_next, diag = super().step(x, u, dt)
        x_next[3] = np.clip(x_next[3], -self.S_MAX, self.S_MAX)
        x_next[4] = np.clip(x_next[4], -self.W_MAX, self.W_MAX)
        return x_next, diag