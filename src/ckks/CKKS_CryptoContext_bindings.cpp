// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "openfhe.h"

#include "ckks/CKKS_key_operations.hpp"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE_CKKS {

void export_CKKS_CryptoContext(py::module_ &m) {

  py::class_<pyOpenFHE_CKKS::CKKSCryptoContext>(m, "CKKSCryptoContext")
      .def(py::init<CryptoContext<DCRTPoly>>())
      .def("enable", &CKKSCryptoContext::enable)
      .def("keyGen", &CKKSCryptoContext::keyGen)
      .def("evalMultKeyGen", &CKKSCryptoContext::evalMultKeyGen)
      .def("evalMultKeysGen", &CKKSCryptoContext::evalMultKeysGen)
      .def("keySwitchGen", &CKKSCryptoContext::keySwitchGen)
      .def("getSchemeID", &CKKSCryptoContext::getSchemeId)
      .def("evalAtIndexKeyGen", &CKKSCryptoContext::evalAtIndexKeyGen1)
      .def("evalAtIndexKeyGen", &CKKSCryptoContext::evalAtIndexKeyGen2)
      .def("evalPowerOf2RotationKeyGen", &CKKSCryptoContext::evalPowerOf2RotationKeyGen)
      .def("evalBootstrapSetup", &CKKSCryptoContext::evalBootstrapSetup)
      .def("evalBootstrapKeyGen", &CKKSCryptoContext::evalBootstrapKeyGen)
      .def("evalBootstrap", &CKKSCryptoContext::evalBootstrap)
      .def("evalBootstrap", &CKKSCryptoContext::evalBootstrapList)
      .def("evalMetaBootstrap", &CKKSCryptoContext::evalMetaBootstrap)
      .def("evalMetaBootstrap", &CKKSCryptoContext::evalMetaBootstrapList)
      .def("encrypt", &CKKSCryptoContext::encryptPublic)
      .def("encrypt", &CKKSCryptoContext::encryptPrivate)
      .def("encrypt", &CKKSCryptoContext::encryptPublic2)
      .def("encrypt", &CKKSCryptoContext::encryptPrivate2)
      .def("decrypt", &CKKSCryptoContext::decrypt)
      .def("getRingDimension", &CKKSCryptoContext::getRingDimension)
      .def("getBatchSize", &CKKSCryptoContext::getBatchSize)
      .def("zeroPadToBatchSize", &CKKSCryptoContext::zeroPadToBatchSizeList)
      .def("zeroPadToBatchSize", &CKKSCryptoContext::zeroPadToBatchSizeNumpy);

  m.def("genCryptoContextCKKS", &genCKKSContext,
        py::arg("multiplicativeDepth"),
        py::arg("scalingFactorBits"),
        py::arg("batchSize"),
        py::arg("stdLevel") = SecurityLevel::HEStd_128_classic,
        py::arg("ringDim") = 0);
}

} // namespace pyOpenFHE_CKKS
