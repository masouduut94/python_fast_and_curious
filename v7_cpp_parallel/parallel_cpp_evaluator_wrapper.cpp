#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "parallel_cpp_evaluator.h"  // Include your parallelized evaluator header

namespace py = pybind11;

PYBIND11_MODULE(parallel_cpp_evaluator, m) {
    py::class_<ParallelCppEvaluator>(m, "ParallelCppEvaluator")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", &ParallelCppEvaluator::evaluate);
}
