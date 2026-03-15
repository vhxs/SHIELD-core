// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include "openfhe.h"

#include <complex>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "ckks/CKKS_key_operations.hpp"
#include "utils/exceptions.hpp"
#include "utils/rotate_utils.hpp"
#include "utils/utils.hpp"

namespace py = pybind11;
using namespace lbcrypto;
using namespace pyOpenFHE;

namespace pyOpenFHE_CKKS {

CKKSCryptoContext CKKSCiphertext::getCryptoContext(void) const {
  return CKKSCryptoContext(cipher->GetCryptoContext());
}

CKKSCiphertext CKKSCiphertext::array_ufunc(
    const py::object &ufunc, const py::object &method,
    const py::array_t<double, py::array::forcecast> &vals,
    const CKKSCiphertext &cipher) {
  std::string op = ufunc.attr("__name__").cast<std::string>();

  if (op == "multiply") {
    return cipher * vals;
  } else if (op == "add") {
    return cipher + vals;
  } else {
    throw pyOpenFHE::not_implemented_exception(
        fmt::format("operator {} between ndarray and CKKSCiphertext", op));
  }
}

bool operator==(const CKKSCiphertext &c1, const CKKSCiphertext &c2) {
  return c1.cipher == c2.cipher;
}

bool operator!=(const CKKSCiphertext &c1, const CKKSCiphertext &c2) {
  return !(c1 == c2);
}

CKKSCiphertext operator+=(CKKSCiphertext &c1, const CKKSCiphertext &c2) {
  c1.cipher += c2.cipher;
  return c1;
}

CKKSCiphertext operator+(CKKSCiphertext c1, const CKKSCiphertext &c2) {
  return c1 += c2;
}

CKKSCiphertext operator-=(CKKSCiphertext &c1, const CKKSCiphertext &c2) {
  c1.cipher -= c2.cipher;
  return c1;
}

CKKSCiphertext operator-(CKKSCiphertext c1, const CKKSCiphertext &c2) {
  return c1 -= c2;
}

CKKSCiphertext operator*=(CKKSCiphertext &c1, const CKKSCiphertext &c2) {
  if ((c1.getTowersRemaining() <= 2) || (c2.getTowersRemaining() <= 2)) {
    throw std::runtime_error(
        fmt::format("Insufficient number of towers remaining to perform a "
                    "multiplication = {}, {}",
                    c1.getTowersRemaining(), c2.getTowersRemaining()));
  }
  c1.cipher *= c2.cipher;
  return c1;
}

CKKSCiphertext operator*(CKKSCiphertext c1, const CKKSCiphertext &c2) {
  return c1 *= c2;
}

CKKSCiphertext EvalRotatePositiveNegativePow2(CKKSCiphertext &ctxt, int r) {
  if (r == 0) {
    return ctxt;
  }
  int N = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  if (abs(r) > N) {
    throw std::runtime_error(fmt::format(
        "rotation value = {} is too large compared to batch size = {}", r, N));
  }
  std::vector<int> po2s = po2Decompose(r);
  for (int i : po2s) {
    ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalAtIndex(ctxt.cipher, i);
  }
  return ctxt;
}

CKKSCiphertext EvalRotatePositivePow2(CKKSCiphertext &ctxt, int r) {
  if (r == 0) {
    return ctxt;
  }
  int N = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  if (abs(r) > N) {
    throw std::runtime_error(fmt::format(
        "rotation value = {} is too large compared to batch size = {}", r, N));
  }
  int mult = (r > 0) ? 1 : -1;
  std::vector<int> po2s = sumOfPo2s(abs(r));
  for (int i : po2s) {
    ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalAtIndex(
        ctxt.cipher, mult * (1 << i));
  }
  return ctxt;
}

CKKSCiphertext CKKSRotateEvalAtIndex(CKKSCiphertext ctxt, int r) {
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalAtIndex(ctxt.cipher, r);
  return ctxt;
}

CKKSCiphertext operator<<=(CKKSCiphertext &ctxt, int r) {
  return EvalRotatePositiveNegativePow2(ctxt, r);
}

py::list CKKSHoistedRotations(const CKKSCiphertext &ctxt, const py::list &pylist) {
  std::vector<int32_t> rotations = pyOpenFHE::pythonListToCppIntVector(pylist);
  auto cc = ctxt.cipher->GetCryptoContext();
  auto cPrecomp = cc->EvalFastRotationPrecompute(ctxt.cipher);
  uint32_t N = cc->GetRingDimension();
  uint32_t M = 2 * N;

  py::list result = pyOpenFHE::make_list(rotations.size());
  for (unsigned int i = 0; i < rotations.size(); i++) {
    result[i] = pyOpenFHE_CKKS::CKKSCiphertext(
        cc->EvalFastRotation(ctxt.cipher, rotations[i], M, cPrecomp));
  }
  return result;
}

CKKSCiphertext operator>>=(CKKSCiphertext &ctxt, double r) {
  return ctxt <<= static_cast<int>(-r);
}

CKKSCiphertext operator<<(CKKSCiphertext ctxt, double r) { return ctxt <<= static_cast<int>(r); }

CKKSCiphertext operator>>(CKKSCiphertext ctxt, double r) { return ctxt >>= r; }

CKKSCiphertext operator+=(CKKSCiphertext &ctxt, double val) {
  std::vector<double> vals = {val};
  size_t dn = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  tileVector(vals, dn);
  auto ptxt = ctxt.cipher->GetCryptoContext()->MakeCKKSPackedPlaintext(vals);
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalAdd(ctxt.cipher, ptxt);
  return ctxt;
}

CKKSCiphertext operator+(CKKSCiphertext ctxt, double val) { return ctxt += val; }
CKKSCiphertext operator+(double val, CKKSCiphertext ctxt) { return ctxt += val; }

CKKSCiphertext operator+=(CKKSCiphertext &ctxt, std::vector<double> vals) {
  size_t N = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  if (vals.size() != N) {
    throw std::runtime_error(
        fmt::format("Provided vector has length = {}, but the CryptoContext batch size = {}",
                    vals.size(), N));
  }
  auto ptxt = ctxt.cipher->GetCryptoContext()->MakeCKKSPackedPlaintext(vals);
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalAdd(ctxt.cipher, ptxt);
  return ctxt;
}

CKKSCiphertext operator+=(CKKSCiphertext &ctxt, const py::list &pyvals) {
  return ctxt += pythonListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator+=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals) {
  return ctxt += numpyListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator+(CKKSCiphertext ctxt, const std::vector<double> &vals) { return ctxt += vals; }
CKKSCiphertext operator+(CKKSCiphertext ctxt, const py::list &vals) { return ctxt += vals; }
CKKSCiphertext operator+(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals) { return ctxt += vals; }
CKKSCiphertext operator+(const std::vector<double> &vals, CKKSCiphertext ctxt) { return ctxt += vals; }
CKKSCiphertext operator+(const py::list &vals, CKKSCiphertext ctxt) { return ctxt += vals; }
CKKSCiphertext operator+(const py::array_t<double, py::array::forcecast> &vals, CKKSCiphertext ctxt) { return ctxt += vals; }

CKKSCiphertext operator-=(CKKSCiphertext &ctxt, std::vector<double> vals) {
  size_t N = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  if (vals.size() != N) {
    throw std::runtime_error(
        fmt::format("Provided vector has length = {}, but the CryptoContext batch size = {}",
                    vals.size(), N));
  }
  auto ptxt = ctxt.cipher->GetCryptoContext()->MakeCKKSPackedPlaintext(vals);
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalSub(ctxt.cipher, ptxt);
  return ctxt;
}

CKKSCiphertext operator-=(CKKSCiphertext &ctxt, const py::list &pyvals) {
  return ctxt -= pythonListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator-=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals) {
  return ctxt -= numpyListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator-(CKKSCiphertext ctxt, const py::list &vals) { return ctxt -= vals; }
CKKSCiphertext operator-(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals) { return ctxt -= vals; }

CKKSCiphertext operator-(const CKKSCiphertext &ctxt) {
  return ctxt.cipher->GetCryptoContext()->EvalNegate(ctxt.cipher);
}

CKKSCiphertext operator-=(CKKSCiphertext &ctxt, double val) {
  ctxt += -val;
  return ctxt;
}

CKKSCiphertext operator-(CKKSCiphertext ctxt, double val) { return ctxt -= val; }
CKKSCiphertext operator-(double val, const CKKSCiphertext &ctxt) {
  auto ncipher = -ctxt;
  return ncipher += val;
}

CKKSCiphertext operator-(CKKSCiphertext ctxt, const std::vector<double> &vals) { return ctxt -= vals; }

CKKSCiphertext operator-(const std::vector<double> &vals, CKKSCiphertext ctxt) {
  return ctxt -= vals;
}

CKKSCiphertext operator-(const py::list &vals, const CKKSCiphertext &ctxt) {
  auto ncipher = -ctxt;
  return ncipher += vals;
}

CKKSCiphertext operator-(const py::array_t<double, py::array::forcecast> &vals, const CKKSCiphertext &ctxt) {
  auto ncipher = -ctxt;
  return ncipher -= vals;
}

CKKSCiphertext CKKSMultiplySingletonIntDoubleAndAdd(const CKKSCiphertext &ctxt, long int val) {
  if (val == 0) return (ctxt - ctxt);
  if (val < 0) return CKKSMultiplySingletonIntDoubleAndAdd(-ctxt, -val);
  CKKSCiphertext doubles = ctxt;
  CKKSCiphertext result = (ctxt - ctxt);
  while (val > 0) {
    if (val & 1) result += doubles;
    doubles = doubles + doubles;
    val >>= 1;
  }
  return result;
}

CKKSCiphertext CKKSMultiplySingletonDirect(CKKSCiphertext ctxt, double val) {
  if (ctxt.getTowersRemaining() <= 2) {
    throw std::runtime_error(
        fmt::format("Insufficient number of towers remaining to perform a "
                    "multiplication = {}", ctxt.getTowersRemaining()));
  }
  std::vector<double> vals = {val};
  size_t dn = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  tileVector(vals, dn);
  auto ptxt = ctxt.cipher->GetCryptoContext()->MakeCKKSPackedPlaintext(vals);
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalMult(ctxt.cipher, ptxt);
  return ctxt;
}

CKKSCiphertext operator*=(CKKSCiphertext &ctxt, long int val) {
  if (abs(val) <= 256) {
    ctxt = CKKSMultiplySingletonIntDoubleAndAdd(ctxt, val);
  } else {
    ctxt = CKKSMultiplySingletonDirect(ctxt, (double)val);
  }
  return ctxt;
}

CKKSCiphertext operator*=(CKKSCiphertext &ctxt, double val) {
  if (ctxt.getTowersRemaining() <= 2) {
    throw std::runtime_error(
        fmt::format("Insufficient number of towers remaining to perform a "
                    "multiplication = {}", ctxt.getTowersRemaining()));
  }
  ctxt = CKKSMultiplySingletonDirect(ctxt, val);
  return ctxt;
}

CKKSCiphertext operator*=(CKKSCiphertext &ctxt, std::vector<double> vals) {
  size_t N = ctxt.cipher->GetEncodingParameters()->GetBatchSize();
  if (vals.size() != N) {
    throw std::runtime_error(
        fmt::format("Provided vector has length = {}, but the CryptoContext batch size = {}",
                    vals.size(), N));
  }
  if (ctxt.getTowersRemaining() <= 2) {
    throw std::runtime_error(
        fmt::format("Insufficient number of towers remaining to perform a "
                    "multiplication = {}", ctxt.getTowersRemaining()));
  }
  auto ptxt = ctxt.cipher->GetCryptoContext()->MakeCKKSPackedPlaintext(vals);
  ctxt.cipher = ctxt.cipher->GetCryptoContext()->EvalMult(ctxt.cipher, ptxt);
  return ctxt;
}

CKKSCiphertext operator*=(CKKSCiphertext &ctxt, const py::list &pyvals) {
  return ctxt *= pythonListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator*=(CKKSCiphertext &ctxt, const py::array_t<double, py::array::forcecast> &pyvals) {
  return ctxt *= numpyListToCppDoubleVector(pyvals);
}

CKKSCiphertext operator*(CKKSCiphertext ctxt, double val) { return ctxt *= val; }
CKKSCiphertext operator*(double val, CKKSCiphertext ctxt) { return ctxt *= val; }
CKKSCiphertext operator*(CKKSCiphertext ctxt, long int val) { return ctxt *= val; }
CKKSCiphertext operator*(long int val, CKKSCiphertext ctxt) { return ctxt *= val; }
CKKSCiphertext operator*(const std::vector<double> &vals, CKKSCiphertext ctxt) { return ctxt *= vals; }
CKKSCiphertext operator*(CKKSCiphertext ctxt, const std::vector<double> &vals) { return ctxt *= vals; }
CKKSCiphertext operator*(CKKSCiphertext ctxt, const py::list &vals) { return ctxt *= vals; }
CKKSCiphertext operator*(const py::list &vals, CKKSCiphertext ctxt) { return ctxt *= vals; }
CKKSCiphertext operator*(CKKSCiphertext ctxt, const py::array_t<double, py::array::forcecast> &vals) { return ctxt *= vals; }
CKKSCiphertext operator*(const py::array_t<double, py::array::forcecast> &vals, CKKSCiphertext ctxt) { return ctxt *= vals; }

} // namespace pyOpenFHE_CKKS
