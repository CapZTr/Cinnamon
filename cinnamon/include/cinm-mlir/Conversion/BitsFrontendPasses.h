/// Declaration of the conversion pass within Bits dialect.
///
/// @file

#pragma once

#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/LinalgToBits/LinalgToBits.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/BitsFrontendPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
