#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <utility>

namespace py = pybind11;

using Vec  = std::vector<double>
using Fn   = std::function<std::pair<Vec, Vec>(const Vec&, const Vec&)

// helper for a + s*b
static Vec axpy(const Vec& a, const Vec& v, double s){
    Vec r(a.size());
    for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + s * b[i];
    return r
}