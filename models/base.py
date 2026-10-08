from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from optim.rk4 import rk4_step
from typing import Protocol

State = np.ndarray[tuple[Any, ...], np.dtype[Any]]
Control = np.ndarray[tuple[Any, ...], np.dtype[Any]]
Diag = dict[str, float]

class ModelBase(ABC):
    """Common interface for any ODE model ẋ = f(x, u)."""
    n_state: int
    n_control: int

    @abstractmethod
    def f(self, x: State, u: Control) -> State:
        """
        Continuous-time dynamics / kinematics: returns ẋ.
        
        Returns (ẋ, diagnostics).
        """

    def step(self, x: State, u: Control, dt: float) -> tuple[State, Diag]:
        """
        Discrete one-step integration.
        """
        return rk4_step(self.f, x, u, dt)

class KinematicBase(ModelBase):
    """
    u is a velocity-level input
    
    Must return a tuple (xdot, diagnositics).
    """
    ...

class DynamicBase(ModelBase):
    """
    u is any input involving forces or movement.

    Must return a tuple (xdot, diagnostics).
    """
    ...


class Controller(Protocol):
    """Anything that maps (t, x) -> u: keyboard, MPC, RL policy, ..."""

    def reset(self) -> None: ...
    def on_key(self) -> None: ...
    def __call__(self, t: float, x: np.ndarray) -> np.ndarray: ...




        