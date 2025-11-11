/// Declaration of the conversion pass within PuD dialect.
///
/// @file

#pragma once

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include "cinm-mlir/Conversion/PuDToFunc/PuDToFunc.h"

namespace mlir {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "cinm-mlir/Conversion/PuDPasses.h.inc"

//===----------------------------------------------------------------------===//

} // namespace mlir
