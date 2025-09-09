/// Implements the Bits dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LogicalResult.h>

#define DEBUG_TYPE "bits-ops"

using namespace mlir;
using namespace mlir::bits;

//===- Generated implementation -------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.cpp.inc"

//===----------------------------------------------------------------------===//
// BitsDialect
//===----------------------------------------------------------------------===//

void BitsDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.cpp.inc"
      >();
}

LogicalResult TransposeOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());
  if (!inputType || inputType.getRank() != 1)
    return emitOpError("input must be a 1D ranked tensor");

  auto outputType = cast<SliceType>(getOutput().getType());
  if (!outputType)
    return emitOpError("output must be of SliceType");

  Type elemType = inputType.getElementType();
  unsigned bitWidth = 0;

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    bitWidth = intTy.getWidth();
  } else if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    bitWidth = floatTy.getWidth();
  } else {
    return emitOpError() << "tensor elements must be integer or float, but got "
                         << elemType;
  }

  if (outputType.getBitWidth() != bitWidth)
    return emitOpError(
        "bit width mismatch between input tensor element and output slice");

  if (outputType.getVectorLength() != inputType.getDimSize(0))
    return emitOpError("vector length mismatch between input and output");

  return success();
}

LogicalResult AssembleOp::verify() {
  auto inputType = cast<SliceType>(getInput().getType());
  if (!inputType)
    return emitOpError("input must be of SliceType");

  auto outputType = cast<RankedTensorType>(getOutput().getType());
  if (!outputType || outputType.getRank() != 1)
    return emitOpError("output must be a 1D ranked tensor");

  Type elemType = outputType.getElementType();
  unsigned bitWidth = 0;

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    bitWidth = intTy.getWidth();
  } else if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    bitWidth = floatTy.getWidth();
  } else {
    return emitOpError() << "tensor elements must be integer or float, but got "
                         << elemType;
  }

  if (inputType.getBitWidth() != bitWidth)
    return emitOpError(
        "bit width mismatch between input slice and output tensor element");

  if (inputType.getVectorLength() != outputType.getDimSize(0))
    return emitOpError("vector length mismatch between input and output");

  return success();
}

LogicalResult SplitSliceVerticallyOp::verify() {
  auto vecLen = getSlice().getType().getVectorLength();
  if (auto constantOp = dyn_cast<arith::ConstantOp>(
      getColumnNum().getDefiningOp())) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      auto rowNum = intAttr.getInt();
      if (vecLen <= rowNum)
        return emitOpError(
            "columnNum of the first slice must be less than the original one");
    } else {
      return emitOpError("columnNum must have explicit value");
    }
  } else {
    return emitOpError("columnNum must have explicit value");
  }

  auto firstColNum = getFirst().getType().getVectorLength();
  auto secondColNum = getSecond().getType().getVectorLength();
  if (firstColNum + secondColNum != vecLen)
    return emitOpError(
        "column's sum of two result slices must be equal to the original one");
  
  auto firstBitwidth = getFirst().getType().getBitWidth();
  auto secondBitwidth = getSecond().getType().getBitWidth();
  auto bitwidth = getSlice().getType().getBitWidth();
  if (firstBitwidth != bitwidth || secondBitwidth != bitwidth)
    return emitOpError(
        "bitwith of two result slices must be equal to the original one");

  return success();
}

LogicalResult MergeSliceVerticallyOp::verify() {
  auto firstBitwidth = getFirst().getType().getBitWidth();
  auto secondBitwidth = getSecond().getType().getBitWidth();
  auto resultBitwidth = getSlice().getType().getBitWidth();
  if (firstBitwidth != resultBitwidth || secondBitwidth != resultBitwidth)
    return emitOpError(
        "bitwith of two input slices must be equal to the result");

  auto firstColNum = getFirst().getType().getVectorLength();
  auto secondColNum = getSecond().getType().getVectorLength();
  auto resultColNum = getSlice().getType().getVectorLength();
  if (firstColNum + secondColNum != resultColNum)
    return emitOpError(
        "column's sum of two input slices must be equal to the result");

  return success();
}

LogicalResult AddIOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() ||
      lhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");

  if (lhsType.getVectorLength() != resType.getVectorLength()
      || rhsType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");
  
  return success();
}

LogicalResult MulFOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() ||
      lhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");

  if (lhsType.getVectorLength() != resType.getVectorLength()
      || rhsType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");
  
  return success();
}

LogicalResult SubOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() ||
      lhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");

  if (lhsType.getVectorLength() != rhsType.getVectorLength() ||
      lhsType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");

  return success();
}
