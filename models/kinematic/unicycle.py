import numpy as np
from models.base import KinematicBase

class Unicycle(KinematicBase):
    n_state, n_control = 3, 2

    def f(self, x, u):
        theta    = x[2]
        v, omega = u
        xdot     = np.array([v*np.cos(theta), v*np.sin(theta), omega])
        diag     = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v [m/s]": v, 
            "omega [rad/s]": omega,
        }
        return xdot, diag