from __future__ import annotations
from typing import Any, Callable

import numpy as np

from models.controllers.pid import PID
from models.base import Controller

Ref = Callable[[float], tuple]


def static_point(p) -> Ref:
    p, z = np.asarray(p, float), np.zeros(2)
    return lambda t: (p, z)

def circle(center, radius, omega, phase=0.0) -> Ref:
    c = np.asarray(center, float)
 
    def ref(t):
        a = phase + omega * t
        return (c + radius * np.array([np.cos(a), np.sin(a)]),
                radius * omega * np.array([-np.sin(a), np.cos(a)]))
    return ref

# Input maps; each model needs is specific input
class UnicycleInput:                      # Unicycle:  u = (v, ω)
    def __call__(self, x, v, w):
        return np.array([v, w])
 
 
class DDRInput:                           # DifferentialDriveRobot:  u = (ωl, ωr)
    def __init__(self, r, L):
        self.r, self.L = r, L
 
    def __call__(self, x, v, w):
        '''DifferentialDriveRobot.f expects wheel angular rates u = (ul, ur)'''
        return np.array([(2*v - w*self.L) / (2*self.r), (2*v + w*self.L) / (2*self.r)])
 
 
class CarInput:                           # SimpleCar:  u = (v, δ)
    def __init__(self, L, delta_max=0.6, v_min=0.2):
        self.L, self.delta_max, self.v_min = L, delta_max, v_min
 
    def steer(self, v, w):
        if abs(v) < 1e-3:
            return float(np.sign(w) * self.delta_max)
        tan_d = self.L * w / v
        return float(np.clip(np.arctan(tan_d), -self.delta_max, self.delta_max))
 
    def __call__(self, x, v, w):
        # Prevent v=0 deadlock when a turning rate w is required
        if abs(v) < 0.05 and abs(w) > 0.01:
            v = self.v_min if v >= 0 else -self.v_min
        return np.array([v, self.steer(v, w)])
 
class BetterUnicycleInput:                # x = [x, y, θ, s],  u = (a, ω)
    def __init__(self, ks):
        self.ks = ks
 
    def __call__(self, x, v, w):
        return np.array([self.ks * (v - x[3]), w])
 
 
class EvenBetterUnicycleInput:            # x = [x, y, θ, s, ω],  u = (a, α)
    def __init__(self, ks, kw):
        self.ks, self.kw = ks, kw
 
    def __call__(self, x, v, w):
        return np.array([self.ks * (v - x[3]), self.kw * (w - x[4])])
 
 
class SmoothDDRInput:                     # x = [x, y, θ, ωl, ωr],  u = (αl, αr)
    def __init__(self, r, L, kw):
        self.wheels, self.kw = DDRInput(r, L), kw
 
    def __call__(self, x, v, w):
        return self.kw * (self.wheels(x, v, w) - x[3:5])
 
 
class ContinuousSteeringCarInput:         # x = [x, y, θ, φ, s],  u = (a, φ̇)
    def __init__(self, L, ks, kphi, delta_max=0.6):
        self.car, self.ks, self.kphi = CarInput(L, delta_max), ks, kphi
 
    def __call__(self, x, v, w):
        phi_des = self.car.steer(v, w)
        return np.array([self.ks * (v - x[4]), self.kphi * (phi_des - x[3])])
 
 
class SteeringAccelCarInput:              # x = [x, y, θ, φ, ω, s],  u = (a, α)
    def __init__(self, L, ks, kphi, kw, delta_max=0.6):
        self.car, self.ks, self.kphi, self.kw = CarInput(L, delta_max), ks, kphi, kw
 
    def __call__(self, x, v, w):
        phi_des = self.car.steer(v, w)
        dphi_des = self.kphi * (phi_des - x[3])
        return np.array([self.ks * (v - x[5]), self.kw * (dphi_des - x[4])])

class SkiddingCarInput:
    def __init__(self, model, delta_max=0.15) -> None:
        L = model.a + model.b
        K = model.m * (model.b / model.cf - model.a / model.cr) / L
        self.gain = (L + K * model.s ** 2) / model.s
        self.delta_max = delta_max

    def __call__(self, x, v, w):
        return np.array([np.clip(self.gain * w, -self.delta_max, self.delta_max)])
 
def input_map(model, ks=4.0, kw=8.0, kphi=4.0):
    n = type(model).__name__
    table = {
        "Unicycle": lambda: UnicycleInput(),
        "BetterUnicycle": lambda: BetterUnicycleInput(ks),
        "EvenBetterUnicycle": lambda: EvenBetterUnicycleInput(ks, kw),
        "DifferentialDriveRobot": lambda: DDRInput(model.r, model.L),
        "SmoothDDR": lambda: SmoothDDRInput(model.r, model.L, kw),
        "SimpleCar": lambda: CarInput(model.L),
        "ContinuousSteeringCar": lambda: ContinuousSteeringCarInput(model.L, ks, kphi),
        "SteeringAccelCar": lambda: SteeringAccelCarInput(model.L, ks, kphi, kw),
        "SkiddingCar": lambda: SkiddingCarInput(model),
    }
    if n not in table:
        raise ValueError(f"no input map for {n}")
    return table[n]()
 
# This is the actual controller, from Astrom (2021), a two-degree-of-freedom controller
class PointTracker(Controller):
    """
    Tracks ref(t) with the point p, d metres ahead of the axle
    """
 
    def __init__(self, model, ref: Ref, pid: PID, d: float = 0.8,
                 v_max: float = np.inf, **gains) -> None:
        self.ref, self.pid, self.d, self.v_max = ref, pid, d, v_max
        self.input_map = input_map(model, **gains)
 
    def reset(self) -> None:
        self.pid.reset()
 
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        th = x[2]
        c, s = np.cos(th), np.sin(th)
        # lookahead point 
        p = x[:2] + self.d * np.array([c, s])
        r, rdot = self.ref(t)

        # 2-DOF PD control: feedforward plus feedback PD (desired velocity vector)
        pdot = rdot + self.pid(t, r - p)
        
        n = np.linalg.norm(pdot)
        if n > self.v_max:
            pdot = pdot * self.v_max / n

        # project pdot onto e_par so we get a scalar v
        v = c * pdot[0] + s * pdot[1]
        # project pdot onto e_perp / d also get a scalar w
        w = (-s * pdot[0] + c * pdot[1]) / self.d
        return self.input_map(x, v, w)