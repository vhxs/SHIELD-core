// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef HE_CNN_POLY_H
#define HE_CNN_POLY_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyOpenFHE_CKKS {
    py::list fhe_gelu(const py::list &py_shards, int degree, double bound);
}

#endif
