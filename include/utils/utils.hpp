// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef OpenFHE_PYTHON_UTILS_H
#define OpenFHE_PYTHON_UTILS_H

#include <complex>
#include <vector>

#include "boost/multi_array.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyOpenFHE {

typedef typename boost::multi_array<double, 2> boost_vector2d;
typedef typename boost::multi_array<double, 4> boost_vector4d;
typedef boost::multi_array_types::index_range srange;
typedef typename boost_vector4d::array_view<4>::type boost_vector4d_slice;

py::list make_list(const std::size_t n, py::object item = py::none());

py::list cppVectorToPythonList(const std::vector<double> &);
py::list cppLongIntVectorToPythonList(const std::vector<int64_t> &vector);
py::array_t<double> cppDoubleVectorToNumpyList(const std::vector<double> &);
py::array_t<int64_t> cppLongIntVectorToNumpyList(const std::vector<int64_t> &vector);
std::vector<int> pythonListToCppIntVector(const py::list &);
std::vector<int> numpyListToCppIntVector(const py::array_t<double, py::array::forcecast> &);
boost_vector4d numpyArrayToCppArray4D(const py::array_t<double, py::array::forcecast> &nplist);
boost_vector2d numpyArrayToCppArray2D(const py::array_t<double, py::array::forcecast> &nplist);
std::vector<double> numpyListToCppDoubleVector(const py::array_t<double, py::array::forcecast> &);
std::vector<double> pythonListToCppDoubleVector(const py::list &);
std::vector<int64_t> pythonListToCppLongIntVector(const py::list &pylist);
std::vector<int64_t> numpyListToCppLongIntVector(const py::array_t<double, py::array::forcecast> &nplist);

} // namespace pyOpenFHE

std::vector<int> sumOfPo2s(int);
void tileVector(std::vector<double> &vals, unsigned int n);
void tileVector(std::vector<int64_t> &vals, unsigned int n);
void tileVector(std::vector<int> &vals, unsigned int n);
template <typename T> void print_vector(std::vector<T> vec);

#endif /* OpenFHE_PYTHON_UTILS_H */
