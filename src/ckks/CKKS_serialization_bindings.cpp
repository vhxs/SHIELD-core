// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <pybind11/pybind11.h>

#include "openfhe.h"

#include "ckks/serialization.hpp"

namespace py = pybind11;

namespace pyOpenFHE_CKKS {

void export_CKKS_serialization(py::module_ &m) {

  py::enum_<pyOpenFHE_CKKS::SerType>(m, "SerType")
      .value("BINARY", pyOpenFHE_CKKS::SerType::BINARY)
      .value("JSON", pyOpenFHE_CKKS::SerType::JSON);

  m.def("SerializeToBytes", SerializeToBytes_Ciphertext);
  m.def("SerializeToBytes", SerializeToBytes_PublicKey);
  m.def("SerializeToBytes", SerializeToBytes_PrivateKey);

  m.def("SerializeToFile", SerializeToFile_Ciphertext);
  m.def("SerializeToFile", SerializeToFile_PublicKey);
  m.def("SerializeToFile", SerializeToFile_PrivateKey);

  m.def("DeserializeFromBytes_Ciphertext", DeserializeFromBytes_Ciphertext);
  m.def("DeserializeFromBytes_PublicKey", DeserializeFromBytes_PublicKey);
  m.def("DeserializeFromBytes_PrivateKey", DeserializeFromBytes_PrivateKey);

  m.def("DeserializeFromFile_Ciphertext", DeserializeFromFile_Ciphertext);
  m.def("DeserializeFromFile_PublicKey", DeserializeFromFile_PublicKey);
  m.def("DeserializeFromFile_PrivateKey", DeserializeFromFile_PrivateKey);

  m.def("SerializeToFile_EvalMultKey_CryptoContext", &SerializeToFile_EvalMultKey_CryptoContext);
  m.def("DeserializeFromFile_CryptoContext", &DeserializeFromFile_CryptoContext);

  m.def("SerializeToFile_EvalMultKey_CryptoContext", &SerializeToFile_EvalMultKey_CryptoContext);
  m.def("SerializeToFile_EvalAutomorphismKey_CryptoContext", &SerializeToFile_EvalAutomorphismKey_CryptoContext);

  m.def("DeserializeFromFile_EvalMultKey_CryptoContext", &DeserializeFromFile_EvalMultKey_CryptoContext);
  m.def("DeserializeFromFile_EvalAutomorphismKey_CryptoContext", &DeserializeFromFile_EvalAutomorphismKey_CryptoContext);

  m.def("SerializeToBytes_EvalMultKey_CryptoContext", &SerializeToBytes_EvalMultKey_CryptoContext);
  m.def("SerializeToBytes_EvalAutomorphismKey_CryptoContext", &SerializeToBytes_EvalAutomorphismKey_CryptoContext);

  m.def("DeserializeFromBytes_EvalMultKey_CryptoContext", &DeserializeFromBytes_EvalMultKey_CryptoContext);
  m.def("DeserializeFromBytes_EvalAutomorphismKey_CryptoContext", &DeserializeFromBytes_EvalAutomorphismKey_CryptoContext);
}

} // namespace pyOpenFHE_CKKS
