/// Implements the PuD dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
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
  auto srcAddrT = cast<RowType>(getSrcAddr().getType());
  auto dstAddrT = cast<RowType>(getDstAddr().getType());

  if (!srcAddrT || !dstAddrT)
    return emitOpError("Operands must be RowType");

  auto srcGroup = srcAddrT.getGroup();
  auto dstGroup = dstAddrT.getGroup();

  if (srcGroup == 0) {
    if (dstGroup == 0 || dstGroup == 2)
      return success();
  } else if (srcGroup == 1) {
    if (dstGroup == 0)
      return success();
  } else if (srcGroup == 2) {
    if (dstGroup == 0 || dstGroup == 2)
      return success();
  }

  return emitOpError("Invalid group of operands");
}

LogicalResult APOp::verify() {
  auto addrT = cast<RowType>(getAddr().getType());

  if (!addrT)
    return emitOpError("Operands must be RowType");

  if (addrT.getGroup() != 0)
    return emitOpError("Operand must be B group");

  return success();
}

LogicalResult StoreOp::verify() {
  auto sliceT = cast<bits::SliceType>(getSlice().getType());
  auto rowT = cast<RowType>(getFirstRow().getType());

  if (!sliceT || !rowT)
    return emitOpError("Invalid operand types");

  if (rowT.getGroup() != 2)
    return emitOpError("First row must in B group");

  return success();
}

LogicalResult LoadOp::verify() {
  auto rowT = cast<RowType>(getFirstRow().getType());
  if (!rowT)
    return emitOpError("First row must in B group");

  auto sliceT = cast<bits::SliceType>(getSlice().getType());
  if (!sliceT)
    return emitOpError("Output must be SliceType");

  return success();
}
