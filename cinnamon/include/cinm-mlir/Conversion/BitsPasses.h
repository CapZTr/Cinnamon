/// Declaration of the conversion pass within Bits dialect.
///
/// @file

#pragma once

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/BitsPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
