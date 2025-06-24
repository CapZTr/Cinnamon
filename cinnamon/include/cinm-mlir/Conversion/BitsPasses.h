/// Declaration of the conversion pass within Bits dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/BitsPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
