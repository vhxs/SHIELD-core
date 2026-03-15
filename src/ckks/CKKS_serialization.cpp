// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#include <stdexcept>
#include <sstream>
#include <fstream>

#include <pybind11/pybind11.h>

#include "ckks/CKKS_ciphertext_extension.hpp"
#include "ckks/CKKS_key_operations.hpp"
#include "ckks/serialization.hpp"
#include "utils/utils.hpp"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

namespace py = pybind11;
using namespace lbcrypto;

namespace pyOpenFHE_CKKS {

py::bytes SerializeToBytes_Ciphertext(const pyOpenFHE_CKKS::CKKSCiphertext &obj,
                                      const pyOpenFHE_CKKS::SerType sertype) {
  std::stringstream ss;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Serialize(obj.cipher, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Serialize(obj.cipher, ss, lbcrypto::SerType::JSON);
  }
  return py::bytes(ss.str());
}

pyOpenFHE_CKKS::CKKSCiphertext
DeserializeFromBytes_Ciphertext(py::bytes py_buffer,
                                const pyOpenFHE_CKKS::SerType sertype) {
  std::string buffer = py_buffer;
  std::stringstream ss(buffer);
  Ciphertext<DCRTPoly> obj;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::JSON);
  }
  return pyOpenFHE_CKKS::CKKSCiphertext(obj);
}

py::bytes SerializeToBytes_PublicKey(const PublicKey<DCRTPoly> &obj,
                                     const pyOpenFHE_CKKS::SerType sertype) {
  std::stringstream ss;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Serialize(obj, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Serialize(obj, ss, lbcrypto::SerType::JSON);
  }
  return py::bytes(ss.str());
}

PublicKey<DCRTPoly>
DeserializeFromBytes_PublicKey(py::bytes py_buffer,
                               const pyOpenFHE_CKKS::SerType sertype) {
  std::string buffer = py_buffer;
  std::stringstream ss(buffer);
  PublicKey<DCRTPoly> obj;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::JSON);
  }
  return obj;
}

py::bytes SerializeToBytes_PrivateKey(const PrivateKey<DCRTPoly> &obj,
                                      const pyOpenFHE_CKKS::SerType sertype) {
  std::stringstream ss;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Serialize(obj, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Serialize(obj, ss, lbcrypto::SerType::JSON);
  }
  return py::bytes(ss.str());
}

PrivateKey<DCRTPoly>
DeserializeFromBytes_PrivateKey(py::bytes py_buffer,
                                const pyOpenFHE_CKKS::SerType sertype) {
  std::string buffer = py_buffer;
  std::stringstream ss(buffer);
  PrivateKey<DCRTPoly> obj;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::BINARY);
  } else {
    Serial::Deserialize(obj, ss, lbcrypto::SerType::JSON);
  }
  return obj;
}

py::bytes SerializeToBytes_EvalMultKey_CryptoContext(
    CKKSCryptoContext &self, const pyOpenFHE_CKKS::SerType sertype) {
  std::stringstream ss;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    self.context->SerializeEvalMultKey(ss, lbcrypto::SerType::BINARY);
  } else {
    self.context->SerializeEvalMultKey(ss, lbcrypto::SerType::JSON);
  }
  return py::bytes(ss.str());
}

bool DeserializeFromBytes_EvalMultKey_CryptoContext(
    CKKSCryptoContext &self, py::bytes py_buffer,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::string buffer = py_buffer;
  std::stringstream ss(buffer);
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    self.context->DeserializeEvalMultKey(ss, lbcrypto::SerType::BINARY);
  } else {
    self.context->DeserializeEvalMultKey(ss, lbcrypto::SerType::JSON);
  }
  return true;
}

py::bytes SerializeToBytes_EvalAutomorphismKey_CryptoContext(
    CKKSCryptoContext &self, const pyOpenFHE_CKKS::SerType sertype) {
  std::stringstream ss;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    self.context->SerializeEvalAutomorphismKey(ss, lbcrypto::SerType::BINARY);
  } else {
    self.context->SerializeEvalAutomorphismKey(ss, lbcrypto::SerType::JSON);
  }
  return py::bytes(ss.str());
}

bool DeserializeFromBytes_EvalAutomorphismKey_CryptoContext(
    CKKSCryptoContext &self, py::bytes py_buffer,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::string buffer = py_buffer;
  std::stringstream ss(buffer);
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    self.context->DeserializeEvalAutomorphismKey(ss, lbcrypto::SerType::BINARY);
  } else {
    self.context->DeserializeEvalAutomorphismKey(ss, lbcrypto::SerType::JSON);
  }
  return true;
}

bool SerializeToFile_Ciphertext(const std::string &filename,
                                const pyOpenFHE_CKKS::CKKSCiphertext &obj,
                                const pyOpenFHE_CKKS::SerType sertype) {
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::SerializeToFile(filename, obj.cipher, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::SerializeToFile(filename, obj.cipher, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not write serialized CKKSCiphertext to file: " + filename);
  }
  return success;
}

bool SerializeToFile_CryptoContext(const std::string &filename,
                                   const CKKSCryptoContext &obj,
                                   const pyOpenFHE_CKKS::SerType sertype) {
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not write serialized CryptoContext to file: " + filename);
  }
  return success;
}

bool SerializeToFile_EvalMultKey_CryptoContext(
    CKKSCryptoContext &self, const std::string &filename,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::ofstream multKeyFile(filename, std::ios::out | std::ios::binary);
  if (!multKeyFile.is_open()) {
    throw std::runtime_error(
        "Could not write serialized EvalMult / relinearization keys to file: " + filename);
  }
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = self.context->SerializeEvalMultKey(multKeyFile, lbcrypto::SerType::BINARY);
  } else {
    success = self.context->SerializeEvalMultKey(multKeyFile, lbcrypto::SerType::JSON);
  }
  multKeyFile.close();
  if (!success) {
    throw std::runtime_error(
        "Could not write serialized EvalMult / relinearization keys to file: " + filename);
  }
  return success;
}

bool SerializeToFile_EvalAutomorphismKey_CryptoContext(
    CKKSCryptoContext &self, const std::string &filename,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::ofstream multKeyFile(filename, std::ios::out | std::ios::binary);
  if (!multKeyFile.is_open()) {
    throw std::runtime_error("Could not write serialized EvalAutomorphism / rotation keys to file: " + filename);
  }
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = self.context->SerializeEvalAutomorphismKey(multKeyFile, lbcrypto::SerType::BINARY);
  } else {
    success = self.context->SerializeEvalAutomorphismKey(multKeyFile, lbcrypto::SerType::JSON);
  }
  multKeyFile.close();
  if (!success) {
    throw std::runtime_error("Could not write serialized EvalAutomorphism / rotation keys to file: " + filename);
  }
  return success;
}

bool SerializeToFile_PublicKey(const std::string &filename,
                               const PublicKey<DCRTPoly> &obj,
                               const pyOpenFHE_CKKS::SerType sertype) {
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not write serialized PublicKey to file: " + filename);
  }
  return success;
}

bool SerializeToFile_PrivateKey(const std::string &filename,
                                const PrivateKey<DCRTPoly> &obj,
                                const pyOpenFHE_CKKS::SerType sertype) {
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::SerializeToFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not write serialized PrivateKey to file: " + filename);
  }
  return success;
}

pyOpenFHE_CKKS::CKKSCiphertext
DeserializeFromFile_Ciphertext(const std::string &filename,
                               const pyOpenFHE_CKKS::SerType sertype) {
  Ciphertext<DCRTPoly> obj;
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not read serialized data from file: " + filename);
  }
  return pyOpenFHE_CKKS::CKKSCiphertext(obj);
}

CryptoContext<DCRTPoly>
DeserializeFromFile_CryptoContext(const std::string &filename,
                                  const pyOpenFHE_CKKS::SerType sertype) {
  throw std::runtime_error(
      "This function is disabled as CryptoContext Deserialization is broken.");
  CryptoContext<DCRTPoly> obj;
  return obj;
}

PublicKey<DCRTPoly>
DeserializeFromFile_PublicKey(const std::string &filename,
                              const pyOpenFHE_CKKS::SerType sertype) {
  PublicKey<DCRTPoly> obj;
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not read serialized data from file: " + filename);
  }
  return obj;
}

PrivateKey<DCRTPoly>
DeserializeFromFile_PrivateKey(const std::string &filename,
                               const pyOpenFHE_CKKS::SerType sertype) {
  PrivateKey<DCRTPoly> obj;
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::BINARY);
  } else {
    success = Serial::DeserializeFromFile(filename, obj, lbcrypto::SerType::JSON);
  }
  if (!success) {
    throw std::runtime_error("Could not read serialized data from file: " + filename);
  }
  return obj;
}

bool DeserializeFromFile_EvalMultKey_CryptoContext(
    CKKSCryptoContext &self, const std::string &filename,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::ifstream multKeyFile(filename, std::ios::in | std::ios::binary);
  if (!multKeyFile.is_open()) {
    throw std::runtime_error(
        "Error reading EvalMult / relinearization keys from file: " + filename);
  }
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = self.context->DeserializeEvalMultKey(multKeyFile, lbcrypto::SerType::BINARY);
  } else {
    success = self.context->DeserializeEvalMultKey(multKeyFile, lbcrypto::SerType::JSON);
  }
  multKeyFile.close();
  return success;
}

bool DeserializeFromFile_EvalAutomorphismKey_CryptoContext(
    CKKSCryptoContext &self, const std::string &filename,
    const pyOpenFHE_CKKS::SerType sertype) {
  std::ifstream multKeyFile(filename, std::ios::in | std::ios::binary);
  if (!multKeyFile.is_open()) {
    throw std::runtime_error(
        "Error reading EvalAutomorphism / rotation keys from file: " + filename);
  }
  bool success = false;
  if (sertype == pyOpenFHE_CKKS::SerType::BINARY) {
    success = self.context->DeserializeEvalAutomorphismKey(multKeyFile, lbcrypto::SerType::BINARY);
  } else {
    success = self.context->DeserializeEvalAutomorphismKey(multKeyFile, lbcrypto::SerType::JSON);
  }
  multKeyFile.close();
  return success;
}

} // namespace pyOpenFHE_CKKS
