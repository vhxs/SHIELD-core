// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef OpenFHE_PYTHON_UTILS_H
#define OpenFHE_PYTHON_UTILS_H

#include <array>
#include <complex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyOpenFHE {

// Owning N-dimensional row-major array backed by a flat std::vector.
// Element access via operator()(i, j, ...), shape via extent(dim).
// TODO: add view() returning std::mdspan once libstdc++ in the build
// environment ships <mdspan> (gcc-toolset-14 on manylinux_2_34 omits it).
template<typename T, std::size_t Rank>
class MDArray {
  std::vector<T> data_;
  std::array<std::size_t, Rank> shape_{};
  std::array<std::size_t, Rank> strides_{};

  void init_strides() {
    strides_[Rank - 1] = 1;
    for (int d = static_cast<int>(Rank) - 2; d >= 0; --d)
      strides_[d] = strides_[d + 1] * shape_[d + 1];
  }

  template<typename... Idx>
  std::size_t linear_index(Idx... idx) const {
    std::array<std::size_t, Rank> idxs{static_cast<std::size_t>(idx)...};
    std::size_t off = 0;
    for (std::size_t d = 0; d < Rank; ++d) off += idxs[d] * strides_[d];
    return off;
  }

public:
  MDArray() = default;

  template<typename... Sizes>
  explicit MDArray(Sizes... sizes)
      : shape_{static_cast<std::size_t>(sizes)...} {
    std::size_t total = 1;
    for (auto s : shape_) total *= s;
    data_.resize(total);
    init_strides();
  }

  std::size_t extent(std::size_t dim) const { return shape_[dim]; }

  template<typename... Idx>
  T& operator()(Idx... idx) { return data_[linear_index(idx...)]; }

  template<typename... Idx>
  const T& operator()(Idx... idx) const { return data_[linear_index(idx...)]; }
};

using array2d = MDArray<double, 2>;
using array4d = MDArray<double, 4>;

py::list make_list(const std::size_t n, py::object item = py::none());

py::list cppVectorToPythonList(const std::vector<double> &);
py::list cppLongIntVectorToPythonList(const std::vector<int64_t> &vector);
py::array_t<double> cppDoubleVectorToNumpyList(const std::vector<double> &);
py::array_t<int64_t> cppLongIntVectorToNumpyList(const std::vector<int64_t> &vector);
std::vector<int> pythonListToCppIntVector(const py::list &);
std::vector<int> numpyListToCppIntVector(const py::array_t<double, py::array::forcecast> &);
array4d numpyArrayToCppArray4D(const py::array_t<double, py::array::forcecast> &nplist);
array2d numpyArrayToCppArray2D(const py::array_t<double, py::array::forcecast> &nplist);
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
