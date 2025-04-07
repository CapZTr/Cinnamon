/// Implements the Bits dialect base.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"

#define DEBUG_TYPE "bits-base"

using namespace mlir;
using namespace mlir::bits;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/Bits/IR/BitsBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BitsDialect
//===----------------------------------------------------------------------===//

void BitsDialect::initialize() {
  registerOps();
  registerTypes();
}
