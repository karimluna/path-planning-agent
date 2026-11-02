'''Adding integrator models actuator inertia.'''

import numpy as np
from models.base import KinematicBase, DynamicBase


class SimpleCar(KinematicBase):
    n_state, n_control = 3, 2
    control_names = ("v", "ω")

    def __init__(self, L: float = 3.0) -> None:
        self.L = L

    def f(self, x, u):
        theta  = x[2]
        v, w   = u
        tan_d = np.tan(w)
        xdot   = np.array([v*np.cos(theta), v*np.sin(theta), (v/self.L)*tan_d])
        diag = {
            "x [m]": x[0],
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v [m/s]": v,
            "omega [rad/s]": v*tan_d/self.L,
            "w [deg]": np.degrees(w),
            "R [m]": self.L / tan_d if abs(tan_d) > 1e-9 else float("inf"),
        }
        return xdot, diag


class ContinuousSteeringCar(KinematicBase):
    n_state, n_control = 5, 2          # x = [x, y, θ, φ, s], u = [u_a, u_ω]
    control_names = ("a", "ω_φ")
    PHI_MAX, S_MAX = 1.0, 3.0

    def __init__(self, L: float = 3.0) -> None:
        self.L = L

    def f(self, x, u):
        theta, phi, s = x[2], x[3], x[4]
        a, u_w = u
        xdot = np.array([s*np.cos(theta), s*np.sin(theta), s*np.tan(phi)/self.L, u_w, a])
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "phi [deg]": np.degrees(phi),
            "v [m/s]": s,
            "a [m/s²]": a,
            "u_w [rad/s]": u_w,
        }
        return xdot, diag

    def step(self, x, u, dt):
        '''Step is modified to clip added integrators.'''
        x_next, diag = super().step(x, u, dt)
        x_next[3] = np.clip(x_next[3], -self.PHI_MAX, self.PHI_MAX)
        x_next[4] = np.clip(x_next[4], -self.S_MAX, self.S_MAX)
        return x_next, diag


class SteeringAccelCar(KinematicBase):
    n_state, n_control = 6, 2          # x = [x, y, θ, φ, ω, s], u = [u_a, u_α]
    control_names = ("a", "α")
    PHI_MAX, S_MAX = 0.89, 3.0

    def __init__(self, L: float = 3.0) -> None:
        self.L = L

    def f(self, x, u):
        theta, phi, w, s = x[2], x[3], x[4], x[5]
        a, alpha = u
        tan_d = np.tan(phi)
        xdot = np.array([s*np.cos(theta), s*np.sin(theta), s*tan_d/self.L, w, alpha, a])
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "phi [deg]": np.degrees(phi),
            "dphi [rad/s]": w,
            "v [m/s]": s,
            "omega [rad/s]": s*tan_d/self.L,
            "a [m/s²]": a,
            "alpha [rad/s²]": alpha,
        }
        return xdot, diag

    def step(self, x, u, dt):
        x_next, diag = super().step(x, u, dt)
        if abs(x_next[3]) >= self.PHI_MAX:          # steering stop
            x_next[3] = np.clip(x_next[3], -self.PHI_MAX, self.PHI_MAX)
            x_next[4] = 0.0
        x_next[5] = np.clip(x_next[5], -self.S_MAX, self.S_MAX)
        return x_next, diag


class SkiddingCar(DynamicBase):
    n_state, n_control = 5, 1

    def __init__(
            self, m: float = 1460, cf: int = 17_000, 
            cr: int = 20_000, a: float = 1.2, b: float = 1.5, 
            I: float = 2170, s: float = 27) -> None:
        self.m  = m     # [kg]
        self.cf = cf    # [-]
        self.cr = cr    # [-]
        self.a  = a     # [m]
        self.b  = b     # [m]
        self.L  = self.a + self.b # [m]
        self.I  = I     # [kg/m²]
        self.s  = s     # [m/s]

    def f(self, x, u): 
        theta, omega, v = x[2], x[3], x[4] # heading, angular velocity, lateral velocity
        u_phi = u[0]
        ff = -self.cf * ((v  + self.a * omega ) / self.s - u_phi) # front tire forces
        fr = -self.cr * (v - self.b * omega) / self.s             # rear tire forces
        xdot = np.array([
            self.s * np.cos(theta) - v * np.sin(theta), 
            self.s * np.sin(theta) + v * np.cos(theta),
            omega,
            (self.a * ff - self.b * fr) / self.I,
            -self.s * omega + (ff + fr) / self.m
        ])
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v [m/s]": np.hypot(self.s, v),
            "omega [rad/s]": omega,
            "beta [deg]": np.degrees(np.arctan2(v, self.s)),   # sideslip angle
            "Ff [N]": ff, "Fr [N]": fr,
        }
        return xdot, diag


class IntegratedSkiddingCar(DynamicBase):
    n_state, n_control = 6, 2 # x = [X, Y, θ, ω, v_y, v_x], u = [u_phi, a_x]

    def __init__(
            self, m: float = 1460, cf: int = 17_000, 
            cr: int = 20_000, a: float = 1.2, b: float = 1.5, 
            I: float = 2170) -> None:
        self.m  = m     
        self.cf = cf    
        self.cr = cr    
        self.a  = a     
        self.b  = b     
        self.L  = self.a + self.b 
        self.I  = I     
        # self.s is removed; velocity is now a state variable.

    def _tire(self, alpha, C, Fz, mu=0.9):   # mu is a new physical parameter
        Fmax = mu * Fz
        return -Fmax * np.tanh(C * alpha / Fmax)

    def f(self, x, u): 
        # Extract states
        theta, omega, v_y, v_x = x[2], x[3], x[4], x[5] 
        # Extract controls
        u_phi, a_x = u[0], u[1]

        # Prevent division by zero when calculating slip angles at low speeds
        v_x_safe = max(v_x, 0.1)

        # Front and rear tire forces (using v_x instead of s)
        ff = self._tire((v_y + self.a*omega)/v_x_safe - u_phi, self.cf, self.m*9.81*self.b/self.L)
        fr = self._tire((v_y - self.b*omega)/v_x_safe, self.cr, self.m*9.81*self.a/self.L)
        
        xdot = np.array([
            v_x * np.cos(theta) - v_y * np.sin(theta),   # X_dot
            v_x * np.sin(theta) + v_y * np.cos(theta),   # Y_dot
            omega,                                       # theta_dot
            (self.a * ff - self.b * fr) / self.I,        # omega_dot
            -v_x * omega + (ff + fr) / self.m,           # v_y_dot (lateral acceleration)
            a_x + v_y * omega                            # v_x_dot (longitudinal accel + Coriolis)
        ])
        
        diag = {
            "x [m]": x[0], 
            "y [m]": x[1],
            "theta [deg]": np.degrees(np.arctan2(np.sin(theta), np.cos(theta))),
            "v_x [m/s]": v_x,
            "v_y [m/s]": v_y,
            "omega [rad/s]": omega,
            "beta [deg]": np.degrees(np.arctan2(v_y, v_x_safe)),
            "Ff [N]": ff, "Fr [N]": fr,
        }
        return xdot, diag