import numpy as np

class KeyboardController:
    """Open-loop teleop: u = ±amp while a key is held, 0 otherwise."""
    KEYMAP = {"up": (0, +1), "down": (0, -1), "left": (1, +1), "right": (1, -1)}

    def __init__(self, n_control: int, amp: float | np.ndarray = 1.0) -> None:
        self.n = n_control
        self.amp = np.broadcast_to(np.asarray(amp, dtype=float), (n_control,)).copy()
        self.pressed: set[str] = set()

    def reset(self) -> None:
        self.pressed.clear()

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        u = np.zeros(self.n)
        for k in self.pressed:
            i, sign = self.KEYMAP[k]
            if i < self.n:
                u[i] += sign * self.amp[i]   # opposite keys cancel
        return u

    def on_key(self, event) -> None:
        if event.key in self.KEYMAP:
            self.pressed.add(event.key)

    def on_key_release(self, event) -> None:
        self.pressed.discard(event.key)