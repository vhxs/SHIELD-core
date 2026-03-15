// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef HE_CNN_POOL_H
#define HE_CNN_POOL_H

#include <vector>

#include <pybind11/pybind11.h>

#include "boost/multi_array.hpp"
#include "utils/utils.hpp"
#include "ckks/CKKS_ciphertext_extension.hpp"

namespace py = pybind11;

namespace pyOpenFHE_CKKS {
    py::list pool(const py::list &py_shards, int mtx_size, bool conv);
}

#endif
