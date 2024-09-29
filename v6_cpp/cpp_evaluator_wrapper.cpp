#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp_evaluator.h"  // Include your evaluator C++ module

namespace py = pybind11;

PYBIND11_MODULE(cpp_evaluator, m) {
    py::class_<CppEvaluator>(m, "CppEvaluator")
        .def(py::init<const std::string&, const std::string&>())  // Constructor binding
        .def("evaluate", &CppEvaluator::evaluate);  // Bind evaluate method
}