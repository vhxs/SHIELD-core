// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <complex>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "openfhe.h"

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "ckks/serialization.hpp"
#include "utils/utils.hpp"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE_CKKS {

void export_CKKS_Ciphertext(py::module_ &m) {

  py::class_<pyOpenFHE_CKKS::CKKSCiphertext>(m, "CKKSCiphertext")
      .def(py::init<Ciphertext<DCRTPoly>>())
      .def(py::init<const pyOpenFHE_CKKS::CKKSCiphertext &>())
      .def(py::init<>())
      .def("getPlaintextModulus", &pyOpenFHE_CKKS::CKKSCiphertext::getPlaintextModulus)
      .def("getScalingFactor", &pyOpenFHE_CKKS::CKKSCiphertext::getScalingFactor)
      .def("getBatchSize", &pyOpenFHE_CKKS::CKKSCiphertext::getBatchSize)
      .def("getMultLevel", &pyOpenFHE_CKKS::CKKSCiphertext::getMultLevel)
      .def("getTowersRemaining", &pyOpenFHE_CKKS::CKKSCiphertext::getTowersRemaining)
      .def("getCryptoContext", &pyOpenFHE_CKKS::CKKSCiphertext::getCryptoContext)
      .def("Rescale", &pyOpenFHE_CKKS::CKKSCiphertext::Rescale, py::arg("levels") = 1)
      .def("compress", &pyOpenFHE_CKKS::CKKSCiphertext::compress)

      .def("RotateEvalAtIndex", &pyOpenFHE_CKKS::CKKSRotateEvalAtIndex)
      .def("HoistedRotations", &pyOpenFHE_CKKS::CKKSHoistedRotations)
      .def("MultiplySingletonDirect", &pyOpenFHE_CKKS::CKKSMultiplySingletonDirect)
      .def("MultiplySingletonIntDoubleAndAdd", &pyOpenFHE_CKKS::CKKSMultiplySingletonIntDoubleAndAdd)

      .def("__copy__", [](const pyOpenFHE_CKKS::CKKSCiphertext &self) {
        return pyOpenFHE_CKKS::CKKSCiphertext(self);
      })
      .def("__eq__",  [](const CKKSCiphertext &a, const CKKSCiphertext &b) { return a == b; })
      .def("__ne__",  [](const CKKSCiphertext &a, const CKKSCiphertext &b) { return a != b; })
      .def("__iadd__", [](CKKSCiphertext &a, const CKKSCiphertext &b) { return a += b; })
      .def("__add__",  [](CKKSCiphertext  a, const CKKSCiphertext &b) { return a +  b; })
      .def("__isub__", [](CKKSCiphertext &a, const CKKSCiphertext &b) { return a -= b; })
      .def("__sub__",  [](CKKSCiphertext  a, const CKKSCiphertext &b) { return a -  b; })
      .def("__imul__", [](CKKSCiphertext &a, const CKKSCiphertext &b) { return a *= b; })
      .def("__mul__",  [](CKKSCiphertext  a, const CKKSCiphertext &b) { return a *  b; })
      // shift (rotation)
      .def("__ilshift__", [](CKKSCiphertext &a, int r) { return a <<= r; })
      .def("__irshift__", [](CKKSCiphertext &a, int r) { return a >>= (double)r; })
      .def("__lshift__",  [](CKKSCiphertext  a, int r) { return a << (double)r; })
      .def("__rshift__",  [](CKKSCiphertext  a, int r) { return a >> (double)r; })
      // scalar double
      .def("__iadd__", [](CKKSCiphertext &a, double v) { return a += v; })
      .def("__add__",  [](CKKSCiphertext  a, double v) { return a +  v; })
      .def("__radd__", [](CKKSCiphertext  a, double v) { return v +  a; })
      .def("__isub__", [](CKKSCiphertext &a, double v) { return a -= v; })
      .def("__sub__",  [](CKKSCiphertext  a, double v) { return a -  v; })
      .def("__rsub__", [](CKKSCiphertext  a, double v) { return v -  a; })
      .def("__imul__", [](CKKSCiphertext &a, double v) { return a *= v; })
      .def("__mul__",  [](CKKSCiphertext  a, double v) { return a *  v; })
      .def("__rmul__", [](CKKSCiphertext  a, double v) { return v *  a; })
      // list operands
      .def("__iadd__", [](CKKSCiphertext &a, const py::list &v) { return a += v; })
      .def("__add__",  [](CKKSCiphertext  a, const py::list &v) { return a +  v; })
      .def("__radd__", [](CKKSCiphertext  a, const py::list &v) { return v +  a; })
      .def("__isub__", [](CKKSCiphertext &a, const py::list &v) { return a -= v; })
      .def("__sub__",  [](CKKSCiphertext  a, const py::list &v) { return a -  v; })
      .def("__rsub__", [](const CKKSCiphertext &a, const py::list &v) { return v - a; })
      .def("__imul__", [](CKKSCiphertext &a, const py::list &v) { return a *= v; })
      .def("__mul__",  [](CKKSCiphertext  a, const py::list &v) { return a *  v; })
      .def("__rmul__", [](CKKSCiphertext  a, const py::list &v) { return v *  a; })
      // numpy array operands
      .def("__iadd__", [](CKKSCiphertext &a, const py::array_t<double, py::array::forcecast> &v) { return a += v; })
      .def("__add__",  [](CKKSCiphertext  a, const py::array_t<double, py::array::forcecast> &v) { return a +  v; })
      .def("__radd__", [](CKKSCiphertext  a, const py::array_t<double, py::array::forcecast> &v) { return v +  a; })
      .def("__isub__", [](CKKSCiphertext &a, const py::array_t<double, py::array::forcecast> &v) { return a -= v; })
      .def("__sub__",  [](CKKSCiphertext  a, const py::array_t<double, py::array::forcecast> &v) { return a -  v; })
      .def("__rsub__", [](const CKKSCiphertext &a, const py::array_t<double, py::array::forcecast> &v) { return v - a; })
      .def("__imul__", [](CKKSCiphertext &a, const py::array_t<double, py::array::forcecast> &v) { return a *= v; })
      .def("__mul__",  [](CKKSCiphertext  a, const py::array_t<double, py::array::forcecast> &v) { return a *  v; })
      .def("__rmul__", [](CKKSCiphertext  a, const py::array_t<double, py::array::forcecast> &v) { return v *  a; })
      .def("__neg__", [](const CKKSCiphertext &a) { return -a; })
      .def("__array_ufunc__", &pyOpenFHE_CKKS::CKKSCiphertext::array_ufunc)
      .def(py::pickle(
          [](const CKKSCiphertext &c) {
            auto data = SerializeToBytes_Ciphertext(c, SerType::JSON);
            return py::make_tuple(data);
          },
          [](py::tuple t) {
            if (t.size() != 1)
              throw std::runtime_error("Invalid pickle state");
            CKKSCiphertext c;
            auto ctxt = DeserializeFromBytes_Ciphertext(t[0].cast<py::bytes>(), SerType::JSON);
            c.cipher = ctxt.cipher;
            return c;
          }));
}

} // namespace pyOpenFHE_CKKS
