// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef HE_CNN_CONV_H
#define HE_CNN_CONV_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "boost/multi_array.hpp"
#include "utils/utils.hpp"

namespace py = pybind11;

namespace pyOpenFHE_CKKS {
    py::list conv2d(const py::list &py_shards, const py::array_t<double, py::array::forcecast> &npfilters, int mtx_size, const py::array_t<double, py::array::forcecast> &permutation);
}

#endif
