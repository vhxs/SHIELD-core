// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef HE_CNN_PYTHON_BINDINGS_H
#define HE_CNN_PYTHON_BINDINGS_H

#include <vector>

#include <pybind11/pybind11.h>

#include "utils/utils.hpp"

namespace py = pybind11;

namespace pyOpenFHE_CKKS {
    using ciphertext_array2d = pyOpenFHE::MDArray<pyOpenFHE_CKKS::CKKSCiphertext, 2>;
    using ciphertext_array4d = pyOpenFHE::MDArray<pyOpenFHE_CKKS::CKKSCiphertext, 4>;
}

#endif
