/// Implements the Bits dialect ops.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
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

  auto elemType = cast<IntegerType>(inputType.getElementType());
  if (!elemType)
    return emitOpError("tensor elements must be integers");

  auto outputType = cast<SliceType>(getOutput().getType());
  if (!outputType)
    return emitOpError("output must be of SliceType");

  if (outputType.getBitWidth() != elemType.getWidth())
    return emitOpError("bit width mismatch between input tensor element and output slice");

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

  auto elemType = cast<IntegerType>(outputType.getElementType());
  if (inputType.getBitWidth() != elemType.getWidth())
    return emitOpError("bit width mismatch between input slice and output tensor element");

  if (inputType.getVectorLength() != outputType.getDimSize(0))
    return emitOpError("vector length mismatch between input and output");

  return success();
}

LogicalResult AddOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() || lhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");

  if (lhsType.getVectorLength() != rhsType.getVectorLength() || lhsType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");

  return success();
}

LogicalResult AddBitOp::verify() {
  auto lhsTy = cast<BitType>(getLhs().getType());
  auto rhsTy = cast<BitType>(getRhs().getType());

  if (!lhsTy || !rhsTy)
    return emitOpError("all operands must be of bits.bit type");

  return mlir::success();
}

LogicalResult SubOp::verify() {
  auto lhsType = cast<SliceType>(getLhs().getType());
  auto rhsType = cast<SliceType>(getRhs().getType());
  auto resType = cast<SliceType>(getResult().getType());

  if (!lhsType || !rhsType || !resType)
    return emitOpError("operands and result must all be of SliceType");

  if (lhsType.getBitWidth() != rhsType.getBitWidth() || lhsType.getBitWidth() != resType.getBitWidth())
    return emitOpError("bit widths of operands and result must match");

  if (lhsType.getVectorLength() != rhsType.getVectorLength() || lhsType.getVectorLength() != resType.getVectorLength())
    return emitOpError("vector lengths of operands and result must match");

  return success();
}

LogicalResult ExtractOp::verify() {
  auto sliceType = cast<SliceType>(getSlice().getType());
  if (!sliceType)
    return emitOpError("slice must be bits.slice type");

  if (!getBitIndex().getType().isInteger(64))
    return emitOpError("bitIndex must be i64");
  if (!getVectorIndex().getType().isInteger(64))
    return emitOpError("vectorIndex must be i64");

  return mlir::success();
}

LogicalResult InsertOp::verify() {
  auto sliceType = cast<SliceType>(getSlice().getType());
  if (!sliceType)
    return emitOpError("slice must be bits.slice type");

  auto bitType = cast<BitType>(getBit().getType());
  if (!bitType)
    return emitOpError("bit must be bits.bit type");

  if (!getBitIndex().getType().isInteger(64))
    return emitOpError("bitIndex must be i64");
  if (!getVectorIndex().getType().isInteger(64))
    return emitOpError("vectorIndex must be i64");

  return mlir::success();
}
