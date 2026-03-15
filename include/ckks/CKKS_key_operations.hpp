// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

// encrypt, decrypt, keygeneration, and the like

#ifndef CKKS_ENCRYPTION_OPENFHE_PYTHON_BINDINGS_H
#define CKKS_ENCRYPTION_OPENFHE_PYTHON_BINDINGS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "utils/utils.hpp"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE_CKKS {
class CKKSCiphertext;

class CKKSCryptoContext {

public:
  CryptoContext<DCRTPoly> context;

  CKKSCryptoContext(CryptoContext<DCRTPoly> cc) : context(cc){};

  CKKSCryptoContext(const CKKSCryptoContext &cc) { context = cc.context; };

  void enable(PKESchemeFeature m) { context->Enable(m); };

  KeyPair<DCRTPoly> keyGen() { return context->KeyGen(); };

  void evalMultKeyGen(const PrivateKey<DCRTPoly> privateKey) {
    context->EvalMultKeyGen(privateKey);
  };
  void evalMultKeysGen(const PrivateKey<DCRTPoly> privateKey) {
    context->EvalMultKeysGen(privateKey);
  };

  EvalKey<DCRTPoly> keySwitchGen(const PrivateKey<DCRTPoly> oldPrivateKey,
                                 const PrivateKey<DCRTPoly> newPrivateKey) {
    return context->KeySwitchGen(oldPrivateKey, newPrivateKey);
  };

  SCHEME getSchemeId() { return context->getSchemeId(); }

  void evalAtIndexKeyGen1(const PrivateKey<DCRTPoly> privateKey,
                          const py::list &index_list) {
    context->EvalAtIndexKeyGen(
        privateKey, pyOpenFHE::pythonListToCppIntVector(index_list));
  };

  void evalAtIndexKeyGen2(const PrivateKey<DCRTPoly> privateKey,
                          const py::array_t<double, py::array::forcecast> &index_list) {
    context->EvalAtIndexKeyGen(
        privateKey, pyOpenFHE::numpyListToCppIntVector(index_list));
  };

  void evalPowerOf2RotationKeyGen(const PrivateKey<DCRTPoly> &);

  void evalBootstrapSetup();
  void evalBootstrapKeyGen(const PrivateKey<DCRTPoly> &);
  py::list evalBootstrapList(py::list);
  pyOpenFHE_CKKS::CKKSCiphertext evalBootstrap(pyOpenFHE_CKKS::CKKSCiphertext);
  py::list evalMetaBootstrapList(py::list);
  pyOpenFHE_CKKS::CKKSCiphertext evalMetaBootstrap(pyOpenFHE_CKKS::CKKSCiphertext);

  Plaintext encode(std::vector<double>);

  pyOpenFHE_CKKS::CKKSCiphertext encryptPrivate(const PrivateKey<DCRTPoly> &,
                                                const py::list &);
  pyOpenFHE_CKKS::CKKSCiphertext encryptPrivate2(const PrivateKey<DCRTPoly> &,
                                                 const py::array_t<double, py::array::forcecast> &);

  pyOpenFHE_CKKS::CKKSCiphertext encryptPublic(const PublicKey<DCRTPoly> &,
                                               const py::list &);
  pyOpenFHE_CKKS::CKKSCiphertext encryptPublic2(const PublicKey<DCRTPoly> &,
                                                const py::array_t<double, py::array::forcecast> &);

  py::array_t<double> decrypt(const PrivateKey<DCRTPoly> &,
                              pyOpenFHE_CKKS::CKKSCiphertext &);

  size_t getBatchSize() {
    return context->GetEncodingParams()->GetBatchSize();
  };

  size_t getRingDimension() { return context->GetRingDimension(); };

  py::array_t<double> zeroPadToBatchSize(std::vector<double>);
  py::array_t<double> zeroPadToBatchSizeList(const py::list &);
  py::array_t<double> zeroPadToBatchSizeNumpy(const py::array_t<double, py::array::forcecast> &);

  template <class Archive> void serialize(Archive &ar) { ar(context); };
};

CKKSCryptoContext genCKKSContext(usint multiplicativeDepth,
                                 usint scalingFactorBits, usint batchSize,
                                 SecurityLevel stdLevel = HEStd_128_classic,
                                 usint ringDim = 0);
} // namespace pyOpenFHE_CKKS

#endif
