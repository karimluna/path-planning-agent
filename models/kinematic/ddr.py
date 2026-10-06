import numpy as np
from models.base import KinematicBase

class DifferentialDriveRobot(KinematicBase):
    n_state, n_control = 3, 2
    control_names = ("ul", "ur")

    def __init__(self, L: float = 0.5, r: float = 0.1) -> None:
        self.L = L
        self.r = r

    def f(self, x, u):
        theta   = x[2]
        ul, ur  = u
        u_omega = (ur + ul) / 2
        u_tsi   = ur - ul
        ruw     = self.r*u_omega
        xdot    = np.array([ruw*np.cos(theta), ruw*np.sin(theta), (self.r/self.L)*u_tsi])
        diag = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "ul [rad/s]": ul,
            "ur [rad/s]": ur,
        }
        return xdot, diag