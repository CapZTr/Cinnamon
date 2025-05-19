/// Implements the PuD dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "pud-ops"

using namespace mlir;
using namespace mlir::pud;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PuDDialect
//===----------------------------------------------------------------------===//

void PuDDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.cpp.inc"
      >();
}

LogicalResult AAPOp::verify() {
  auto srcAddrType = cast<RowAddressType>(getSrcAddr().getType());
  auto dstAddrType = cast<RowAddressType>(getDstAddr().getType());

  if (!srcAddrType || !dstAddrType)
    return emitOpError("Addresses of AAP must be RowAddressType");

  return success();
}

LogicalResult APOp::verify() {
  auto addrType = cast<RowAddressType>(getAddr().getType());

  if (!addrType)
    return emitOpError("Address of AAP must be RowAddressType");

  return success();
}
