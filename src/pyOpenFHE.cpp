// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <pybind11/pybind11.h>

#include "ckks/bindings.hpp"
#include "utils/enums_binding.hpp"
#include "utils/exceptions.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyOpenFHE, m) {

  py::register_exception<pyOpenFHE::not_implemented_exception>(
      m, "NotImplementedException", PyExc_NotImplementedError);
  py::register_exception<pyOpenFHE::type_exception>(
      m, "TypeException", PyExc_TypeError);

  auto enums_m = m.def_submodule("enums");
  pyOpenFHE::export_enums(enums_m);

  auto CKKS_m = m.def_submodule("CKKS");
  pyOpenFHE_CKKS::export_CKKS_CryptoContext(CKKS_m);
  pyOpenFHE_CKKS::export_CKKS_Ciphertext(CKKS_m);

  auto serial_m = CKKS_m.def_submodule("serial");
  pyOpenFHE_CKKS::export_CKKS_serialization(serial_m);

  auto CNN_m = CKKS_m.def_submodule("CNN");
  pyOpenFHE_CKKS::export_he_cnn_functions(CNN_m);
}
