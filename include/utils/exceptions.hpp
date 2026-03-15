// (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

#ifndef OpenFHE_PYTHON_EXCEPTIONS_H
#define OpenFHE_PYTHON_EXCEPTIONS_H

#include <stdexcept>

namespace pyOpenFHE {

class not_implemented_exception : public std::logic_error {
  using std::logic_error::logic_error;
};

class type_exception : public std::logic_error {
  using std::logic_error::logic_error;
};

} // namespace pyOpenFHE

#endif /* OpenFHE_PYTHON_EXCEPTIONS_H */
