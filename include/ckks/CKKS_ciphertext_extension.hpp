// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef CKKS_OPENFHE_PYTHON_CIPHERTEXT_H
#define CKKS_OPENFHE_PYTHON_CIPHERTEXT_H

#include <complex>
#include <vector>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "openfhe.h"

#include "ckks/CKKS_key_operations.hpp"
#include "utils/utils.hpp"

#include "constants.h"
#include "encoding/encodingparams.h"
#include "scheme/scheme-utils.h"
#include "utils/exceptions.hpp"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE_CKKS {

class CKKSCryptoContext;

class CKKSCiphertext {

public:
  Ciphertext<DCRTPoly> cipher;

  CKKSCiphertext(){};

  CKKSCiphertext(Ciphertext<DCRTPoly> cipher) : cipher(cipher){};

  CKKSCiphertext(const CKKSCiphertext &ctxt) {
    cipher = Ciphertext<DCRTPoly>(ctxt.cipher);
  };

  CKKSCiphertext array_ufunc(const py::object &ufunc,
                             const py::object &method,
                             const py::array_t<double, py::array::forcecast> &vals,
                             const CKKSCiphertext &cipher);

  uint64_t getPlaintextModulus(void) const {
    return cipher->GetCryptoContext()
        ->GetEncodingParams()
        ->GetPlaintextModulus();
  };

  CKKSCryptoContext getCryptoContext(void) const;

  double getScalingFactor(void) const {
    return cipher->GetScalingFactor();
  };

  uint64_t getBatchSize(void) const {
    return cipher->GetCryptoContext()->GetEncodingParams()->GetBatchSize();
  };

  uint64_t getMultLevel(void) const { return cipher->GetLevel(); };

  uint64_t getTowersRemaining(void) const {
    const std::vector<DCRTPoly> &cv = cipher->GetElements();
    usint sizeQl = cv[0].GetNumOfElements();
    return sizeQl;
  }

  CKKSCiphertext compress(size_t towersLeft) const {
    if (getTowersRemaining() <= towersLeft) {
      throw std::runtime_error(fmt::format("Cannot compress to {} towers, towers remaining = {}", towersLeft, getTowersRemaining()));
    }
    auto cipher2 = cipher->GetCryptoContext()->Compress(cipher, towersLeft);
    return CKKSCiphertext(cipher2);
  }

  CKKSCiphertext Rescale(size_t levels = 1) const {
    if (getTowersRemaining() <= 1 + levels) {
      throw std::runtime_error(
          fmt::format("Insufficient number of towers remaining = {} to perform "
                      "{} rescalings",
                      getTowersRemaining(), levels));
    }
    auto algo = cipher->GetCryptoContext()->GetScheme();
    auto cipher2 = algo->ModReduce(cipher, levels);
    return CKKSCiphertext(cipher2);
  }
};

bool operator==(const CKKSCiphertext &c1, const CKKSCiphertext &c2);
bool operator!=(const CKKSCiphertext &c1, const CKKSCiphertext &c2);
CKKSCiphertext operator+=(CKKSCiphertext &c1, const CKKSCiphertext &c2);
CKKSCiphertext operator+(CKKSCiphertext c1, const CKKSCiphertext &c2);
CKKSCiphertext operator-=(CKKSCiphertext &c1, const CKKSCiphertext &c2);
CKKSCiphertext operator-(CKKSCiphertext c1, const CKKSCiphertext &c2);
CKKSCiphertext operator*=(CKKSCiphertext &c1, const CKKSCiphertext &c2);
CKKSCiphertext operator*(CKKSCiphertext c1, const CKKSCiphertext &c2);
CKKSCiphertext operator<<=(CKKSCiphertext &ctxt, int r);
CKKSCiphertext CKKSRotateEvalAtIndex(CKKSCiphertext ctxt, int r);
py::list CKKSHoistedRotations(const CKKSCiphertext &ctxt, const py::list &pylist);
CKKSCiphertext CKKSMultiplySingletonDirect(CKKSCiphertext ctxt, double val);
CKKSCiphertext CKKSMultiplySingletonIntDoubleAndAdd(const CKKSCiphertext &ctxt, long int val);
CKKSCiphertext operator>>=(CKKSCiphertext &ctxt, double r);
CKKSCiphertext operator<<(CKKSCiphertext ctxt, double r);
CKKSCiphertext operator>>(CKKSCiphertext ctxt, double r);
CKKSCiphertext operator+=(CKKSCiphertext &ctxt, double val);
CKKSCiphertext operator+(CKKSCiphertext ctxt, double val);
CKKSCiphertext operator+(double val, CKKSCiphertext ctxt);
CKKSCiphertext operator+=(CKKSCiphertext &ctxt, std::vector<double> vals);
CKKSCiphertext operator+=(CKKSCiphertext &ctxt, const py::list &pyvals);
CKKSCiphertext operator+=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals);
CKKSCiphertext operator+(CKKSCiphertext ctxt, const std::vector<double> &vals);
CKKSCiphertext operator+(const std::vector<double> &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator+(CKKSCiphertext ctxt, const py::list &vals);
CKKSCiphertext operator+(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals);
CKKSCiphertext operator+(const py::list &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator+(const py::array_t<double, py::array::forcecast> &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator-=(CKKSCiphertext &ctxt, std::vector<double> vals);
CKKSCiphertext operator-=(CKKSCiphertext &ctxt, const py::list &pyvals);
CKKSCiphertext operator-=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals);
CKKSCiphertext operator-(CKKSCiphertext ctxt, const py::list &vals);
CKKSCiphertext operator-(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals);
CKKSCiphertext operator-(const CKKSCiphertext &ctxt);
CKKSCiphertext operator-=(CKKSCiphertext &ctxt, double val);
CKKSCiphertext operator-(CKKSCiphertext ctxt, double val);
CKKSCiphertext operator-(double val, const CKKSCiphertext &ctxt);
CKKSCiphertext operator-(CKKSCiphertext ctxt, const std::vector<double> &vals);
CKKSCiphertext operator-(const std::vector<double> &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator-(const py::list &vals, const CKKSCiphertext &ctxt);
CKKSCiphertext operator-(const py::array_t<double, py::array::forcecast> &vals, const CKKSCiphertext &ctxt);
CKKSCiphertext operator*=(CKKSCiphertext &ctxt, std::vector<double> vals);
CKKSCiphertext operator*=(CKKSCiphertext &ctxt, double val);
CKKSCiphertext operator*=(CKKSCiphertext &ctxt, long int val);
CKKSCiphertext operator*=(CKKSCiphertext &ctxt, const py::list &pyvals);
CKKSCiphertext operator*=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals);
CKKSCiphertext operator*(CKKSCiphertext ctxt, double val);
CKKSCiphertext operator*(double val, CKKSCiphertext ctxt);
CKKSCiphertext operator*(CKKSCiphertext ctxt, long int val);
CKKSCiphertext operator*(long int val, CKKSCiphertext ctxt);
CKKSCiphertext operator*(CKKSCiphertext ctxt, const std::vector<double> &vals);
CKKSCiphertext operator*(const std::vector<double> &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator*(CKKSCiphertext ctxt, const py::list &vals);
CKKSCiphertext operator*(const py::list &vals, CKKSCiphertext ctxt);
CKKSCiphertext operator*(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals);
CKKSCiphertext operator*(const py::array_t<double, py::array::forcecast> &vals, CKKSCiphertext ctxt);

} // namespace pyOpenFHE_CKKS

#endif /* CKKS_OPENFHE_PYTHON_CIPHERTEXT_H */
