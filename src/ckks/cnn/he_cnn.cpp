// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <pybind11/pybind11.h>

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "ckks/bindings.hpp"
#include "ckks/cnn/he_cnn.hpp"
#include "ckks/cnn/pool.hpp"
#include "ckks/cnn/upsample.hpp"
#include "ckks/cnn/conv.hpp"
#include "ckks/cnn/linear.hpp"
#include "ckks/cnn/poly.hpp"
#include <omp.h>

namespace py = pybind11;
using namespace pyOpenFHE;
using namespace pyOpenFHE_CKKS;

void pyOpenFHE_CKKS::export_he_cnn_functions(py::module_ &m) {
    m.def("conv2d", conv2d);
    m.def("linear", linear);
    m.def("pool", pool);
    m.def("upsample", upsample);
    m.def("fhe_gelu", fhe_gelu);
    m.def("omp_set_num_threads", omp_set_num_threads);
    m.def("omp_set_nested", omp_set_nested);
    m.def("omp_set_dynamic", omp_set_dynamic);
}
