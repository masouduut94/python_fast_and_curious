#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For automatic conversion of STL containers
#include "shared_mutex_parallel_evaluator.h"

namespace py = pybind11;

// Define the bindings
PYBIND11_MODULE(shared_mutex_parallel_evaluator, m) {
    // Bind the SharedMutexParallelCppEvaluator class
    py::class_<SharedMutexParallelCppEvaluator>(m, "SharedMutexParallelCppEvaluator")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", &SharedMutexParallelCppEvaluator::evaluate);
}
