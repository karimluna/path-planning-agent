#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <vector>
#include <utility>

namespace py = pybind11;

using Vec = std::vector<double>;
using Fn  = std::function<std::pair<Vec, py::object>(const Vec&, const Vec&)>;

// helper for a + s*b
static Vec axpy(const Vec& a, const Vec& v, double s) {
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        r[i] = a[i] + s * v[i];
    }
    return r;
}

// Fourth-order Runge-Kutta step
std::pair<Vec, py::object> rungeKutta4Step(
    const Fn& fun,
    const Vec& state,
    const Vec& controls,
    double dt = 1.0
) {
    // first step returns dk1 and diagnostics
    auto step1 = fun(state, controls);
    const Vec& k1   = step1.first;
    py::object diag = step1.second;

    // intermediate evaluation steps
    Vec k2 = fun(axpy(state, k1, 0.5 * dt), controls).first;
    Vec k3 = fun(axpy(state, k2, 0.5 * dt), controls).first;
    Vec k4 = fun(axpy(state, k3, dt), controls).first;

    // compute final integrated state
    Vec next_state(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        next_state[i] = state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    return {next_state, diag};
}


PYBIND11_MODULE(runge_kutta, m, py::mod_gil_not_used()) {
    m.doc() = "High performance RK4 numerical integration module";

    m.def("rk4_step", &rungeKutta4Step,
    py::arg("fun"),
    py::arg("state"),
    py::arg("controls"),
    py::arg("dt") = 1.0,
    R"doc(
    Performs one step of RK4 integration.
          
    Returns (next_state      
    diagnostics).
    )doc");
}

