import numpy as np

def rk4_step(
        fun, 
        state: np.ndarray, 
        controls: np.ndarray, 
        dt: float = 1.0,
    ):
    """
    Performs one step of RK4 integration.
    
    Returns next state and diagnostics vector.
    """
    k1, diag = fun(state, controls)
    k2, _    = fun(state + 0.5 * dt * k1, controls)
    k3, _    = fun(state + 0.5 * dt * k2, controls)
    k4, _    = fun(state + dt * k3, controls)
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state, diag
