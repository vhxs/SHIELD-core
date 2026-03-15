// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <stdexcept>

#include <fmt/format.h>

#include <pybind11/pybind11.h>

#include "ckks/CKKS_key_operations.hpp"
#include "openfhe.h"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE {

void export_enums(py::module_ &m) {

  py::class_<PublicKeyImpl<DCRTPoly>, std::shared_ptr<PublicKeyImpl<DCRTPoly>>>(m, "PublicKey")
      .def("getCryptoContext",
           [](PublicKeyImpl<DCRTPoly> &self) -> pyOpenFHE_CKKS::CKKSCryptoContext {
             return pyOpenFHE_CKKS::CKKSCryptoContext(self.GetCryptoContext());
           });

  py::class_<PrivateKeyImpl<DCRTPoly>, std::shared_ptr<PrivateKeyImpl<DCRTPoly>>>(m, "PrivateKey")
      .def("getCryptoContext",
           [](PrivateKeyImpl<DCRTPoly> &self) -> pyOpenFHE_CKKS::CKKSCryptoContext {
             return pyOpenFHE_CKKS::CKKSCryptoContext(self.GetCryptoContext());
           });

  py::class_<KeyPair<DCRTPoly>>(m, "KeyPair")
      .def(py::init<PublicKey<DCRTPoly>, PrivateKey<DCRTPoly>>())
      .def_readonly("publicKey", &KeyPair<DCRTPoly>::publicKey)
      .def_readonly("secretKey", &KeyPair<DCRTPoly>::secretKey)
      .def("good", &KeyPair<DCRTPoly>::good);

  py::class_<EvalKeyImpl<DCRTPoly>, std::shared_ptr<EvalKeyImpl<DCRTPoly>>>(m, "EvalKey");

  py::enum_<PKESchemeFeature>(m, "PKESchemeFeature")
      .value("PKE", PKESchemeFeature::PKE)
      .value("KEYSWITCH", PKESchemeFeature::KEYSWITCH)
      .value("PRE", PKESchemeFeature::PRE)
      .value("LEVELEDSHE", PKESchemeFeature::LEVELEDSHE)
      .value("ADVANCEDSHE", PKESchemeFeature::ADVANCEDSHE)
      .value("MULTIPARTY", PKESchemeFeature::MULTIPARTY)
      .value("FHE", PKESchemeFeature::FHE);

  py::enum_<SecretKeyDist>(m, "SecretKeyDist")
      .value("GAUSSIAN", SecretKeyDist::GAUSSIAN)
      .value("UNIFORM_TERNARY", SecretKeyDist::UNIFORM_TERNARY)
      .value("SPARSE_TERNARY", SecretKeyDist::SPARSE_TERNARY);

  py::enum_<ScalingTechnique>(m, "ScalingTechnique")
      .value("FIXEDMANUAL", ScalingTechnique::FIXEDMANUAL)
      .value("FIXEDAUTO", ScalingTechnique::FIXEDAUTO)
      .value("FLEXIBLEAUTO", ScalingTechnique::FLEXIBLEAUTO)
      .value("FLEXIBLEAUTOEXT", ScalingTechnique::FLEXIBLEAUTOEXT)
      .value("NORESCALE", ScalingTechnique::NORESCALE)
      .value("INVALID_RS_TECHNIQUE", ScalingTechnique::INVALID_RS_TECHNIQUE);

  py::enum_<SecurityLevel>(m, "SecurityLevel")
      .value("HEStd_128_classic", SecurityLevel::HEStd_128_classic)
      .value("HEStd_192_classic", SecurityLevel::HEStd_192_classic)
      .value("HEStd_256_classic", SecurityLevel::HEStd_256_classic)
      .value("HEStd_NotSet", SecurityLevel::HEStd_NotSet);

  py::enum_<EncryptionTechnique>(m, "EncryptionTechnique")
      .value("STANDARD", EncryptionTechnique::STANDARD)
      .value("EXTENDED", EncryptionTechnique::EXTENDED);

  py::enum_<KeySwitchTechnique>(m, "KeySwitchTechnique")
      .value("INVALID_KS_TECH", KeySwitchTechnique::INVALID_KS_TECH)
      .value("BV", KeySwitchTechnique::BV)
      .value("HYBRID", KeySwitchTechnique::HYBRID);

  py::enum_<MultiplicationTechnique>(m, "MultiplicationTechnique")
      .value("BEHZ", MultiplicationTechnique::BEHZ)
      .value("HPS", MultiplicationTechnique::HPS)
      .value("HPSPOVERQ", MultiplicationTechnique::HPSPOVERQ)
      .value("HPSPOVERQLEVELED", MultiplicationTechnique::HPSPOVERQLEVELED);

  py::enum_<LargeScalingFactorConstants>(m, "LargeScalingFactorConstants")
      .value("MAX_BITS_IN_WORD", LargeScalingFactorConstants::MAX_BITS_IN_WORD)
      .value("MAX_LOG_STEP", LargeScalingFactorConstants::MAX_LOG_STEP);

  py::enum_<SCHEME>(m, "SCHEME")
      .value("INVALID_SCHEME", SCHEME::INVALID_SCHEME)
      .value("CKKSRNS_SCHEME", SCHEME::CKKSRNS_SCHEME)
      .value("BFVRNS_SCHEME", SCHEME::BFVRNS_SCHEME)
      .value("BGVRNS_SCHEME", SCHEME::BGVRNS_SCHEME);
}

} // namespace pyOpenFHE
