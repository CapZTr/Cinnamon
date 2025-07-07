/// Implements the PuD dialect base.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"

#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#define DEBUG_TYPE "pud-base"

using namespace mlir;
using namespace mlir::pud;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/PuD/IR/PuDBase.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PuDDialect
//===----------------------------------------------------------------------===//

void PuDDialect::initialize() {
  registerOps();
  registerTypes();
}