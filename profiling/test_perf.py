import time
import numpy as np
from guolao import runge_kutta


class DoubleIntegrator:
    n_state, n_control = 2, 1
    control_names = ("u",)

    def f(self, x, u):
        p, v = x  # [position, velocity]
        a = float(np.asarray(u).reshape(-1)[0])
        xdot = [v, a]
        diag = {
            "p [m/s]": p,
            "v [m/s]": v,
            "a [m/s²]": a,
        }
        return xdot, diag

    def pystep(self, x, u, dt):
        # Pure Python implementation of RK4 step
        k1, diag = self.f(x, u)
        k1 = np.array(k1)
        k2, _ = self.f(x + 0.5 * dt * k1, u)
        k2 = np.array(k2)
        k3, _ = self.f(x + 0.5 * dt * k2, u)
        k3 = np.array(k3)
        k4, _ = self.f(x + dt * k3, u)
        k4 = np.array(k4)
        
        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next, diag

    def cppstep(self, x, u, dt):
        x_next, diag = runge_kutta.rk4_step(self.f, x, u, dt)
        return np.array(x_next), diag


def test_perf():
    model = DoubleIntegrator()
    x0 = np.array([0.0, 1.0])
    u0 = np.array([0.5])
    dt = 0.01
    n_steps = 10_000

    # correctness Verification
    x_py, diag_py = model.pystep(x0, u0, dt)
    x_cpp, diag_cpp = model.cppstep(x0, u0, dt)
    
    assert np.allclose(x_py, x_cpp), f"Mismatch detected: Py {x_py} vs C++ {x_cpp}"
    assert diag_py == diag_cpp, "Diagnostic dict mismatch"
    print("Numerical test passed: C++ and Python outputs match.\n")

    # benchmark Python RK4
    start_py = time.perf_counter()
    x_curr = x0.copy()
    for _ in range(n_steps):
        x_curr, _ = model.pystep(x_curr, u0, dt)
    py_elapsed = time.perf_counter() - start_py

    # benchmark C++ RK4
    start_cpp = time.perf_counter()
    x_curr = x0.copy()
    for _ in range(n_steps):
        x_curr, _ = model.cppstep(x_curr, u0, dt)
    cpp_elapsed = time.perf_counter() - start_cpp

    # output benchmark summary
    print(f"--- Benchmark Results ({n_steps:,} steps) ---")
    print(f"Python step time: {py_elapsed * 1000:.2f} ms ({n_steps / py_elapsed:.0f} steps/s)")
    print(f"C++ step time:    {cpp_elapsed * 1000:.2f} ms ({n_steps / cpp_elapsed:.0f} steps/s)")
    print(f"Speedup ratio:    {py_elapsed / cpp_elapsed:.2f}x")


if __name__ == "__main__":
    test_perf()