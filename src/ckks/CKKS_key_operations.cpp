// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <complex>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "ckks/CKKS_key_operations.hpp"
#include "ckks/serialization.hpp"
#include "utils/utils.hpp"

namespace py = pybind11;
using namespace lbcrypto;
using namespace pyOpenFHE;

namespace pyOpenFHE_CKKS {

CKKSCryptoContext genCKKSContext(usint multiplicativeDepth,
                                 usint scalingFactorBits, usint batchSize,
                                 SecurityLevel stdLevel, usint ringDim) {
  CCParams<CryptoContextCKKSRNS> parameters;
  parameters.SetMultiplicativeDepth(multiplicativeDepth);
  parameters.SetScalingModSize(scalingFactorBits);
  parameters.SetBatchSize(batchSize);
  parameters.SetSecurityLevel(stdLevel);
  if (ringDim != 0) {
    parameters.SetRingDim(ringDim);
  }
  CryptoContext<DCRTPoly> native_cc = GenCryptoContext(parameters);
  return CKKSCryptoContext(native_cc);
}

Plaintext CKKSCryptoContext::encode(std::vector<double> vals) {
  if (vals.size() != context->GetEncodingParams()->GetBatchSize()) {
    std::string s =
        fmt::format("Provided vector has length = {}, but the CryptoContext "
                    "batch size = {}",
                    vals.size(), context->GetEncodingParams()->GetBatchSize());
    throw std::runtime_error(s);
  }
  std::vector<std::complex<double>> cvals(vals.size());
  for (unsigned int i = 0; i < cvals.size(); i++) {
    cvals[i] = vals[i];
  }
  return context->MakeCKKSPackedPlaintext(cvals);
}

pyOpenFHE_CKKS::CKKSCiphertext
CKKSCryptoContext::encryptPrivate(const PrivateKey<DCRTPoly> &privateKey,
                                  const py::list &pyvals) {
  std::vector<double> vals = pyOpenFHE::pythonListToCppDoubleVector(pyvals);
  auto ptxt = encode(vals);
  return pyOpenFHE_CKKS::CKKSCiphertext(context->Encrypt(privateKey, ptxt));
}

pyOpenFHE_CKKS::CKKSCiphertext
CKKSCryptoContext::encryptPublic(const PublicKey<DCRTPoly> &publicKey,
                                 const py::list &pyvals) {
  std::vector<double> vals = pyOpenFHE::pythonListToCppDoubleVector(pyvals);
  auto ptxt = encode(vals);
  return pyOpenFHE_CKKS::CKKSCiphertext(context->Encrypt(publicKey, ptxt));
}

pyOpenFHE_CKKS::CKKSCiphertext
CKKSCryptoContext::encryptPrivate2(const PrivateKey<DCRTPoly> &privateKey,
                                   const py::array_t<double, py::array::forcecast> &pyvals) {
  std::vector<double> vals = pyOpenFHE::numpyListToCppDoubleVector(pyvals);
  auto ptxt = encode(vals);
  return pyOpenFHE_CKKS::CKKSCiphertext(context->Encrypt(privateKey, ptxt));
}

pyOpenFHE_CKKS::CKKSCiphertext
CKKSCryptoContext::encryptPublic2(const PublicKey<DCRTPoly> &publicKey,
                                  const py::array_t<double, py::array::forcecast> &pyvals) {
  std::vector<double> vals = pyOpenFHE::numpyListToCppDoubleVector(pyvals);
  auto ptxt = encode(vals);
  return pyOpenFHE_CKKS::CKKSCiphertext(context->Encrypt(publicKey, ptxt));
}

py::array_t<double> CKKSCryptoContext::zeroPadToBatchSize(std::vector<double> vals) {
  size_t batch_size = context->GetEncodingParams()->GetBatchSize();
  if (vals.size() > batch_size) {
    std::string s =
        fmt::format("Provided vector has length = {}, but the CryptoContext "
                    "batch size = {}",
                    vals.size(), context->GetEncodingParams()->GetBatchSize());
    throw std::runtime_error(s);
  }
  vals.resize(batch_size, 0);
  return pyOpenFHE::cppDoubleVectorToNumpyList(vals);
}

py::array_t<double> CKKSCryptoContext::zeroPadToBatchSizeList(const py::list &pyvals) {
  std::vector<double> vals = pyOpenFHE::pythonListToCppDoubleVector(pyvals);
  return zeroPadToBatchSize(vals);
}

py::array_t<double> CKKSCryptoContext::zeroPadToBatchSizeNumpy(const py::array_t<double, py::array::forcecast> &pyvals) {
  std::vector<double> vals = pyOpenFHE::numpyListToCppDoubleVector(pyvals);
  return zeroPadToBatchSize(vals);
}

py::array_t<double> CKKSCryptoContext::decrypt(const PrivateKey<DCRTPoly> &privateKey,
                                               pyOpenFHE_CKKS::CKKSCiphertext &ctxt) {
  Plaintext ptxt;
  auto ctxt2 = ctxt.cipher->GetCryptoContext()->Compress(ctxt.cipher, 2);
  context->Decrypt(privateKey, ctxt2, &ptxt);
  ptxt->SetLength(ctxt.cipher->GetEncodingParameters()->GetBatchSize());
  auto cvals = ptxt->GetRealPackedValue();
  std::vector<double> vals(cvals.size());
  for (unsigned int i = 0; i < cvals.size(); i++) {
    vals[i] = std::real(cvals[i]);
  }
  return pyOpenFHE::cppDoubleVectorToNumpyList(vals);
}

void CKKSCryptoContext::evalBootstrapSetup() {
  std::vector<uint32_t> bsgsDim = {0, 0};
  std::vector<uint32_t> levelBudget = {4, 4};
  usint slots = context->GetEncodingParams()->GetBatchSize();
  context->EvalBootstrapSetup(levelBudget, bsgsDim, slots);
}

void CKKSCryptoContext::evalBootstrapKeyGen(const PrivateKey<DCRTPoly> &privateKey) {
  usint slots = context->GetEncodingParams()->GetBatchSize();
  context->EvalBootstrapKeyGen(privateKey, slots);
}

py::list CKKSCryptoContext::evalBootstrapList(py::list ctxts) {
  int n = static_cast<int>(ctxts.size());
  std::vector<pyOpenFHE_CKKS::CKKSCiphertext> input_ctxts(n);
  auto output_ctxts = pyOpenFHE::make_list(n);

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    input_ctxts[i] = ctxts[i].cast<pyOpenFHE_CKKS::CKKSCiphertext>();
    input_ctxts[i].cipher = context->EvalBootstrap(input_ctxts[i].cipher);
    output_ctxts[i] = input_ctxts[i];
  }
  return output_ctxts;
}

pyOpenFHE_CKKS::CKKSCiphertext CKKSCryptoContext::evalMetaBootstrap(pyOpenFHE_CKKS::CKKSCiphertext ctxt) {
  double error_scale = 1e-3;
  auto c2 = pyOpenFHE_CKKS::CKKSCiphertext(context->EvalBootstrap(ctxt.cipher));
  auto e1 = (ctxt - c2) * (1 / error_scale);
  auto e2 = pyOpenFHE_CKKS::CKKSCiphertext(context->EvalBootstrap(e1.cipher)) * error_scale;
  return c2 + e2;
}

py::list CKKSCryptoContext::evalMetaBootstrapList(py::list ctxts) {
  int n = static_cast<int>(ctxts.size());
  std::vector<pyOpenFHE_CKKS::CKKSCiphertext> input_ctxts(n);
  auto output_ctxts = pyOpenFHE::make_list(n);

  for (int i = 0; i < n; ++i) {
    input_ctxts[i] = ctxts[i].cast<pyOpenFHE_CKKS::CKKSCiphertext>();
  }

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    input_ctxts[i] = evalMetaBootstrap(input_ctxts[i]);
  }

  for (int i = 0; i < n; ++i) {
    output_ctxts[i] = input_ctxts[i];
  }
  return output_ctxts;
}

pyOpenFHE_CKKS::CKKSCiphertext CKKSCryptoContext::evalBootstrap(pyOpenFHE_CKKS::CKKSCiphertext ctxt) {
  ctxt.cipher = context->EvalBootstrap(ctxt.cipher);
  return ctxt;
}

void CKKSCryptoContext::evalPowerOf2RotationKeyGen(const PrivateKey<DCRTPoly> &privateKey) {
  int N = context->GetEncodingParams()->GetBatchSize();
  int M = context->GetRingDimension();
  N = std::min(N, M / 2);
  std::vector<int> index_list;
  int r = 1;
  while (r <= N) {
    index_list.push_back(r);
    index_list.push_back(-r);
    r *= 2;
  }
  context->EvalAtIndexKeyGen(privateKey, index_list);
}

} // namespace pyOpenFHE_CKKS
