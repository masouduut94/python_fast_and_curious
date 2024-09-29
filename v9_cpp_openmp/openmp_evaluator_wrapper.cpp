#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // This allows automatic conversion of C++ STL types like std::vector to Python lists
#include "openmp_evaluator.h"

namespace py = pybind11;

PYBIND11_MODULE(openmp_evaluator, m) {
    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<int, int, int, int, int, int, int>())
        .def("calculate_iou", &BoundingBox::calculate_iou)
        .def("is_true_positive_or_false_positive", &BoundingBox::is_true_positive_or_false_positive)
        .def("is_false_negative", &BoundingBox::is_false_negative);

    py::class_<OpenmpEvaluator>(m, "OpenmpEvaluator")
        .def(py::init<const std::string&, const std::string&>())
        .def("evaluate", &OpenmpEvaluator::evaluate);
}
