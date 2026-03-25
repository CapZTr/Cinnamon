/// Implements the Bits dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <cstdint>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
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
  if (!inputType || (inputType.getRank() != 1 && inputType.getRank() != 2))
    return emitOpError("input must be a 1D/2D ranked tensor");

  int64_t outputBitWidth = 0;
  if (inputType.getRank() == 1) {
    auto outputType = cast<SliceType>(getOutput().getType());
    if (!outputType)
      return emitOpError("output must be of SliceType if input is 1D");
    outputBitWidth = outputType.getBitWidth();
  } else {
    auto outputType = cast<CubeType>(getOutput().getType());
    if (!outputType)
      return emitOpError("output must be of CubeType if input is 2D");
    outputBitWidth = outputType.getBitWidth();
  }

  Type elemType = inputType.getElementType();
  int64_t bitWidth = 0;

  if (auto intTy = dyn_cast<IntegerType>(elemType)) {
    bitWidth = intTy.getWidth();
  } else if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    bitWidth = floatTy.getWidth();
  } else {
    return emitOpError() << "tensor elements must be integer or float, but got "
                         << elemType;
  }

  if (outputBitWidth != bitWidth)
    return emitOpError(
        "bit width mismatch between input tensor element and output slice");

  return success();
}

LogicalResult AssembleOp::verify() {
  auto outputType = cast<RankedTensorType>(getOutput().getType());
  if (!outputType || (outputType.getRank() != 1 && outputType.getRank() != 2))
    return emitOpError("output must be a 1D/2D ranked tensor");

  int64_t inputBitWidth = 0;
  if (outputType.getRank() == 1) {
    auto inputType = cast<SliceType>(getInput().getType());
    if (!inputType)
      return emitOpError("input must be of SliceType if output is 1D tensor");
    inputBitWidth = inputType.getBitWidth();
  } else {
    auto inputType = cast<CubeType>(getInput().getType());
    if (!inputType)
      return emitOpError("input must be of CubeType if output is 2D tensor");
    inputBitWidth = inputType.getBitWidth();
  }

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

  if (inputBitWidth != bitWidth)
    return emitOpError(
        "bit width mismatch between input slice and output tensor element");

  return success();
}

LogicalResult ExtractSliceOp::verify() {
  if (getCube().getType().getBitWidth() != getSlice().getType().getBitWidth() ||
      getCube().getType().getVectorLength() != getSlice().getType().getVectorLength()) {
    return emitOpError("bitwidth and vectorlength of cube and slice must match");
  }

  return success();
}

LogicalResult InsertSliceOp::verify() {
  if (getCube().getType().getBitWidth() != getSlice().getType().getBitWidth() ||
      getCube().getType().getVectorLength() != getSlice().getType().getVectorLength()) {
    return emitOpError("bitwidth and vectorlength of cube and slice must match");
  }
  
  return success();
}

LogicalResult ExtensionIOp::verify() {
  // auto operandBitWidth = getSlice().getType().getBitWidth();
  // auto resultBitWidth = getResult().getType().getBitWidth();
  // if (auto constantOp = dyn_cast<arith::ConstantOp>(getRowNumToExt().getDefiningOp())) {
  //   if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
  //     auto rowNum = intAttr.getInt();
  //     if (rowNum + operandBitWidth != resultBitWidth) {
  //       return emitOpError("rowNum doesn't match after extension");
  //     }
  //   } else {
  //     return emitOpError("rowNumToExt must have explicit value");
  //   }
  // } else {
  //   return emitOpError("rowNumToExt must be constant");
  // }

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

LogicalResult MulIOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() + rhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("sum of bit widths of operands and result must match");

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

LogicalResult AndOp::verify() {
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

LogicalResult OrOp::verify() {
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

LogicalResult XOrOp::verify() {
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

LogicalResult MajOp::verify() {
  auto lhsType = cast<BitRowType>(getLhs().getType());
  auto rhsType = cast<BitRowType>(getRhs().getType());
  auto thirdType = cast<BitRowType>(getThird().getType());
  auto resType = cast<BitRowType>(getResult().getType());

  if (!lhsType || !rhsType || !thirdType || !resType)
    return emitOpError("operands and result must all be of BitRowType");

  if (lhsType.getVectorLength() != resType.getVectorLength() ||
      rhsType.getVectorLength() != resType.getVectorLength() ||
      thirdType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");

  return success();
}

LogicalResult NotOp::verify() {
  auto inputType = cast<BitRowType>(getInput().getType());
  auto resultType = cast<BitRowType>(getResult().getType());

  if (!inputType || !resultType)
    return emitOpError("input and result must be of BitRowType");

  if (inputType.getVectorLength() != resultType.getVectorLength())
    return emitOpError("vector lengths of input and result must match");

  return success();
}

LogicalResult MaxOp::verify() {
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

LogicalResult MinOp::verify() {
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

LogicalResult MatmulIOp::verify() {
  auto lhsType = cast<CubeType>(getLhs().getType());
  auto rhsType = cast<CubeType>(getRhs().getType());
  auto resType = cast<CubeType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of CubeType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() ||
      lhsType.getBitWidth() * 2 != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");
  
  if (lhsType.getVectorLength() != resType.getVectorLength() ||
      rhsType.getHeight() != resType.getHeight() ||
      lhsType.getHeight() != rhsType.getVectorLength())
    return emitOpError("wrong size");

  return success();
}

LogicalResult MatvecIOp::verify() {
  auto lhsType = cast<CubeType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("wrong type");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() ||
      lhsType.getBitWidth() * 2 != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");
  
  if (lhsType.getVectorLength() != resType.getVectorLength() ||
      lhsType.getHeight() != rhsType.getVectorLength())
    return emitOpError("wrong size");

  return success();
}
