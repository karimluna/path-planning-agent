import numpy as np

class KeyboardController:
    """Open-loop teleoperation."""

    def __init__(self, n_control: int, du: float | np.ndarray = 0.25) -> None:
        self.u = np.zeros(n_control)
        self.du = np.broadcast_to(np.asarray(du, dtype=float), (n_control,)).copy()

    def reset(self) -> None:
        self.u[:] = 0.0

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.u

    def on_key(self, event) -> None:
        n = len(self.u)
        if event.key == "up":
            self.u[0] += self.du[0]
        elif event.key == "down":
            self.u[0] -= self.du[0]
        elif event.key == "left" and n > 1:
            self.u[1] += self.du[1]
        elif event.key == "right" and n > 1:
            self.u[1] -= self.du[1]
        elif event.key == " ":
            self.u[:] = 0.0