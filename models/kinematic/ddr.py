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
            "v [m/s]": self.r * (ur + ul)/ 2,
            "omega [rad/s]": self.r*(ur-ul)/self.L,
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "ul [rad/s]": ul,
            "ur [rad/s]": ur,
        }
        return xdot, diag


class SmoothDDR(KinematicBase):
    n_state, n_control = 5, 2
    control_names = ("ul", "ur")

    def __init__(self, L: float = 0.5, r: float = 0.1) -> None:
        self.L = L
        self.r = r

    def f(self, x, u):
        wr      = x[4]
        wl      = x[3]
        theta   = x[2]
        ul, ur  = u
        xdot    = np.array([self.r/2*(wl + wr)*np.cos(theta), self.r/2*(wl + wr)*np.sin(theta), self.r/self.L*(wr - wl), ul, ur])

        diag = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "wl [rad/s]": wl,
            "wr [rad/s]": wr,
            "al [rad/s]": ul,
            "ar [rad/s]": ur,
            "v [m/s]": self.r / 2 * (wl + wr),
            "omega [rad/s]": self.r/self.L*(wr-wl),
        }
        return xdot, diag