from models.base import Controller
import numpy as np

class PID:
    """Vector PID."""
    def __init__(self, kp, ki=0.0, kd=0.0, i_max=np.inf) -> None:
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self.reset()
 
    def reset(self) -> None:
        self.i, self.e_prev, self.t_prev = 0.0, None, None
 
    def __call__(self, t: float, e: np.ndarray) -> np.ndarray:
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        out = self.kp * e
        if dt > 0.0:
            self.i = np.clip(self.i + e * dt, -self.i_max, self.i_max)   # clamp
            if self.e_prev is not None:
                out = out + self.kd * (e - self.e_prev) / dt
        out = out + self.ki * self.i
        self.e_prev, self.t_prev = e.copy(), t
        return out