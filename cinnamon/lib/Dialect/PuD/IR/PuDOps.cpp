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

  // A row held at a constant can be read into anything; what nothing may do is
  // write into one, which is what rejecting a constant destination says. Seeding
  // a data row from the zero row is how a carry that lives in a row starts, so
  // constant-to-data is as ordinary as constant-to-compute.
  if (dstGroup != 1)
    return success();

  return emitOpError("Invalid group of operands");
}

LogicalResult RowCopyOp::verify() {
  auto srcAddrT = cast<RowType>(getSrcAddr().getType());
  auto dstAddrT = cast<RowType>(getDstAddr().getType());

  if (!srcAddrT || !dstAddrT)
    return emitOpError("Operands must be RowType");

  // A copy moves data between rows that hold it. Neither end may be a control
  // row: a compute row is not somewhere a value waits, and a constant row is
  // not somewhere anything is written.
  if (srcAddrT.getGroup() == 1 || dstAddrT.getGroup() == 1)
    return emitOpError("a row copy moves data rows, not control rows");

  return success();
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
