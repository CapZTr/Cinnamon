/// Declaration of the transform pass within Bits dialect.
///
/// @file

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir::bits {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir::bits
