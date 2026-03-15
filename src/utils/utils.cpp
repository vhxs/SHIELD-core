// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "utils/utils.hpp"

namespace py = pybind11;

py::list pyOpenFHE::make_list(const std::size_t n, py::object item) {
  py::list result;
  for (std::size_t i = 0; i < n; ++i) {
    result.append(item);
  }
  return result;
}

py::list pyOpenFHE::cppVectorToPythonList(const std::vector<double> &vector) {
  py::list pythonList;
  for (unsigned int i = 0; i < vector.size(); i++) {
    pythonList.append(vector[i]);
  }
  return pythonList;
}

py::list pyOpenFHE::cppLongIntVectorToPythonList(const std::vector<int64_t> &vector) {
  py::list pythonList;
  for (unsigned int i = 0; i < vector.size(); i++) {
    pythonList.append(vector[i]);
  }
  return pythonList;
}

py::array_t<double> pyOpenFHE::cppDoubleVectorToNumpyList(const std::vector<double> &vector) {
  py::array_t<double> result(vector.size());
  auto buf = result.request();
  double *ptr = static_cast<double *>(buf.ptr);
  for (unsigned int i = 0; i < vector.size(); i++) {
    ptr[i] = vector[i];
  }
  return result;
}

py::array_t<int64_t> pyOpenFHE::cppLongIntVectorToNumpyList(const std::vector<int64_t> &vector) {
  py::array_t<int64_t> result(vector.size());
  auto buf = result.request();
  int64_t *ptr = static_cast<int64_t *>(buf.ptr);
  for (unsigned int i = 0; i < vector.size(); i++) {
    ptr[i] = vector[i];
  }
  return result;
}

std::vector<int> pyOpenFHE::pythonListToCppIntVector(const py::list &pylist) {
  std::vector<int> cppVector;
  for (unsigned int i = 0; i < pylist.size(); i++) {
    cppVector.push_back(pylist[i].cast<int>());
  }
  return cppVector;
}

std::vector<int> pyOpenFHE::numpyListToCppIntVector(const py::array_t<double, py::array::forcecast> &nplist) {
  auto buf = nplist.request();
  double *ptr = static_cast<double *>(buf.ptr);
  std::vector<int> cppVector(buf.shape[0]);
  for (ssize_t i = 0; i < buf.shape[0]; i++) {
    cppVector[i] = static_cast<int>(ptr[i]);
  }
  return cppVector;
}

std::vector<int64_t> pyOpenFHE::pythonListToCppLongIntVector(const py::list &pylist) {
  std::vector<int64_t> cppVector;
  for (unsigned int i = 0; i < pylist.size(); i++) {
    cppVector.push_back(pylist[i].cast<int64_t>());
  }
  return cppVector;
}

std::vector<int64_t> pyOpenFHE::numpyListToCppLongIntVector(const py::array_t<double, py::array::forcecast> &nplist) {
  auto buf = nplist.request();
  double *ptr = static_cast<double *>(buf.ptr);
  std::vector<int64_t> cppVector(buf.shape[0]);
  for (ssize_t i = 0; i < buf.shape[0]; i++) {
    cppVector[i] = static_cast<int64_t>(ptr[i]);
  }
  return cppVector;
}

std::vector<double> pyOpenFHE::numpyListToCppDoubleVector(const py::array_t<double, py::array::forcecast> &nplist) {
  auto buf = nplist.request();
  if (buf.ndim != 1) {
    throw std::runtime_error(
        fmt::format("Numpy array must be one-dimensional but had dimension: {}", buf.ndim));
  }
  double *ptr = static_cast<double *>(buf.ptr);
  return std::vector<double>(ptr, ptr + buf.shape[0]);
}

pyOpenFHE::boost_vector2d pyOpenFHE::numpyArrayToCppArray2D(const py::array_t<double, py::array::forcecast> &nplist) {
  auto buf = nplist.request();
  if (buf.ndim != 2) {
    throw std::runtime_error(fmt::format(
        "Numpy array must be two-dimensional but had dimension: {}", buf.ndim));
  }
  double *ptr = static_cast<double *>(buf.ptr);
  pyOpenFHE::boost_vector2d cppVector(boost::extents[buf.shape[0]][buf.shape[1]]);
  for (ssize_t i0 = 0; i0 < buf.shape[0]; i0++) {
    for (ssize_t i1 = 0; i1 < buf.shape[1]; i1++) {
      cppVector[i0][i1] = ptr[i0 * buf.shape[1] + i1];
    }
  }
  return cppVector;
}

pyOpenFHE::boost_vector4d pyOpenFHE::numpyArrayToCppArray4D(const py::array_t<double, py::array::forcecast> &nplist) {
  auto buf = nplist.request();
  if (buf.ndim != 4) {
    throw std::runtime_error(fmt::format(
        "Numpy array must be four-dimensional but had dimension: {}", buf.ndim));
  }
  double *ptr = static_cast<double *>(buf.ptr);
  pyOpenFHE::boost_vector4d cppVector(boost::extents[buf.shape[0]][buf.shape[1]][buf.shape[2]][buf.shape[3]]);
  for (ssize_t i0 = 0; i0 < buf.shape[0]; i0++) {
    for (ssize_t i1 = 0; i1 < buf.shape[1]; i1++) {
      for (ssize_t i2 = 0; i2 < buf.shape[2]; i2++) {
        for (ssize_t i3 = 0; i3 < buf.shape[3]; i3++) {
          cppVector[i0][i1][i2][i3] = ptr[i0 * buf.strides[0]/sizeof(double)
                                        + i1 * buf.strides[1]/sizeof(double)
                                        + i2 * buf.strides[2]/sizeof(double)
                                        + i3];
        }
      }
    }
  }
  return cppVector;
}

std::vector<double> pyOpenFHE::pythonListToCppDoubleVector(const py::list &pylist) {
  std::vector<double> cppVector;
  for (unsigned int i = 0; i < pylist.size(); i++) {
    cppVector.push_back(pylist[i].cast<double>());
  }
  return cppVector;
}

void tileVector(std::vector<double> &vals, unsigned int final_size) {
  auto current_size = vals.size();
  vals.resize(final_size);
  while (current_size < final_size) {
    std::copy_n(vals.begin(), current_size, vals.begin() + current_size);
    current_size <<= 1;
  }
}

void tileVector(std::vector<int64_t> &vals, unsigned int final_size) {
  auto current_size = vals.size();
  vals.resize(final_size);
  while (current_size < final_size) {
    std::copy_n(vals.begin(), current_size, vals.begin() + current_size);
    current_size <<= 1;
  }
}

void tileVector(std::vector<int> &vals, unsigned int final_size) {
  auto current_size = vals.size();
  vals.resize(final_size);
  while (current_size < final_size) {
    std::copy_n(vals.begin(), current_size, vals.begin() + current_size);
    current_size <<= 1;
  }
}

template <typename T> void print_vector(std::vector<T> vec) {
  for (int i = 0; i < (int)vec.size(); i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
}
