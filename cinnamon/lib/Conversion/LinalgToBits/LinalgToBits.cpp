#include "cinm-mlir/Conversion/LinalgToBits/LinalgToBits.h"

#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>

using namespace mlir;
using namespace mlir::bits;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/BitsFrontendPasses.h.inc>

namespace {

struct InputCache {
  static llvm::DenseMap<Value, Value> &get() {
    static llvm::DenseMap<Value, Value> cache;
    return cache;
  }
};

struct SliceCache {
  static llvm::DenseMap<Operation *, Value> &get() {
    static llvm::DenseMap<Operation *, Value> cache;
    return cache;
  }
};

static bool isExtSIGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return false;

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto ext = dyn_cast_or_null<arith::ExtSIOp>(
      yield.getValues().front().getDefiningOp());
  if (!ext)
    return false;

  auto arg = dyn_cast<BlockArgument>(ext.getIn());
  return arg && arg.getOwner() == &gBlock && arg.getArgNumber() == 0;
}

static bool isTruncIGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return false;

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto trunc = dyn_cast_or_null<arith::TruncIOp>(
      yield.getValues().front().getDefiningOp());
  if (!trunc)
    return false;

  auto arg = dyn_cast<BlockArgument>(trunc.getIn());
  return arg && arg.getOwner() == &gBlock && arg.getArgNumber() == 0;
}

static Value stripExtSIGeneric(Value tensorValue) {
  auto generic =
      dyn_cast_or_null<linalg::GenericOp>(tensorValue.getDefiningOp());
  if (!generic || generic.getInputs().size() != 1 ||
      generic.getOutputs().size() != 1)
    return {};

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return {};

  auto ext = dyn_cast_or_null<arith::ExtSIOp>(
      yield.getValues().front().getDefiningOp());
  if (!ext)
    return {};

  auto arg = dyn_cast<BlockArgument>(ext.getIn());
  if (!arg || arg.getOwner() != &gBlock || arg.getArgNumber() != 0)
    return {};

  return generic.getInputs().front();
}

static Value getSourceTensorBeforeExt(Value tensorValue) {
  if (Value source = stripExtSIGeneric(tensorValue))
    return source;
  return tensorValue;
}

static void eraseExtSIGenericIfDead(Value tensorValue,
                                    ConversionPatternRewriter &rewriter) {
  auto generic =
      dyn_cast_or_null<linalg::GenericOp>(tensorValue.getDefiningOp());
  if (!generic || !isExtSIGeneric(generic) || !generic->use_empty())
    return;
  rewriter.eraseOp(generic);
}

static Value createTransposeFromTensor(Value tensor,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc, MLIRContext *ctx) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType)
    return {};

  auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
  if (!elementType)
    return {};

  if (tensorType.getRank() == 1) {
    auto outputType =
        SliceType::get(ctx, elementType.getWidth(), tensorType.getShape()[0]);
    return rewriter.create<TransposeOp>(loc, outputType, tensor);
  }

  if (tensorType.getRank() == 2) {
    auto outputType =
        CubeType::get(ctx, elementType.getWidth(), tensorType.getShape()[0],
                      tensorType.getShape()[1]);
    return rewriter.create<TransposeOp>(loc, outputType, tensor);
  }

  return {};
}

static RankedTensorType inferTensorTypeFromBitsValue(Value bitsValue,
                                                     MLIRContext *ctx) {
  if (auto sliceType = dyn_cast<SliceType>(bitsValue.getType())) {
    auto elemType = IntegerType::get(ctx, sliceType.getBitWidth());
    return RankedTensorType::get({sliceType.getVectorLength()}, elemType);
  }

  if (auto cubeType = dyn_cast<CubeType>(bitsValue.getType())) {
    auto elemType = IntegerType::get(ctx, cubeType.getBitWidth());
    return RankedTensorType::get(
        {cubeType.getVectorLength(), cubeType.getHeight()}, elemType);
  }

  return {};
}

static Value getOrCreateIndexConstant(ConversionPatternRewriter &rewriter,
                                      Location loc, int64_t value) {
  auto parentFunc = rewriter.getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<func::FuncOp>();
  if (parentFunc) {
    Block &entry = parentFunc.front();
    for (Operation &entryOp : entry) {
      auto cst = dyn_cast<arith::ConstantIndexOp>(entryOp);
      if (cst && cst.value() == value)
        return cst.getResult();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entry);
    return rewriter.create<arith::ConstantIndexOp>(loc, value);
  }

  return rewriter.create<arith::ConstantIndexOp>(loc, value);
}

struct ConvertLinalgGenericToBits
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumLoops() != 1)
      return failure();

    for (AffineMap map : op.getIndexingMapsArray()) {
      if (!map.isIdentity())
        return failure();
    }

    for (utils::IteratorType iteratorType : op.getIteratorTypesArray()) {
      if (iteratorType != utils::IteratorType::parallel)
        return failure();
    }

    if (op.getOutputs().size() != 1)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!outputType || outputType.getRank() != 1)
      return failure();

    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    int64_t vectorLen = outputType.getShape()[0];

    auto &inputCache = InputCache::get();
    auto &sliceCache = SliceCache::get();

    auto getOrCreateSlice = [&](Value operand) -> Value {
      Operation *defOp = operand.getDefiningOp();
      if (defOp) {
        auto it = sliceCache.find(defOp);
        if (it != sliceCache.end())
          return it->second;
      }

      auto it = inputCache.find(operand);
      if (it != inputCache.end())
        return it->second;

      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType)
        return {};

      auto rank = operandType.getRank();
      if (rank != 1 && rank != 2) {
        return {};
      }

      auto operandElemType = operandType.getElementType();
      int64_t operandBitWidth = operandElemType.getIntOrFloatBitWidth();
      Value slice;
      if (rank == 1) {
        auto operandSliceType = SliceType::get(ctx, operandBitWidth, vectorLen);
        slice = rewriter.create<TransposeOp>(loc, operandSliceType, operand);
      } else {
      }

      inputCache[operand] = slice;

      return slice;
    };

    auto getMulOperandSlice = [&](Value tensorOperand) -> Value {
      if (auto source = stripExtSIGeneric(tensorOperand)) {
        return getOrCreateSlice(source);
      }

      if (auto assemble = dyn_cast_or_null<bits::AssembleOp>(
              tensorOperand.getDefiningOp())) {
        if (auto ext = dyn_cast_or_null<bits::ExtensionIOp>(
                assemble.getInput().getDefiningOp())) {
          return ext.getSlice();
        }
      }

      return getOrCreateSlice(tensorOperand);
    };

    auto &block = op.getRegion().front();
    if (block.getOperations().size() != 2)
      return failure();

    auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1)
      return failure();

    Value yieldedValue = yieldOp.getValues().front();
    Operation *yieldedOp = yieldedValue.getDefiningOp();
    if (!yieldedOp)
      return failure();

    if (op.getInputs().size() == 1) {
      auto extOp = dyn_cast<arith::ExtSIOp>(yieldedOp);
      if (!extOp)
        return failure();

      auto arg = dyn_cast<BlockArgument>(extOp.getIn());
      if (!arg || arg.getOwner() != &block || arg.getArgNumber() != 0)
        return failure();

      auto inputType =
          dyn_cast<RankedTensorType>(op.getInputs().front().getType());
      if (!inputType || inputType.getRank() != 1)
        return failure();

      auto inputElemType = dyn_cast<IntegerType>(inputType.getElementType());
      auto outputElemType = dyn_cast<IntegerType>(outputType.getElementType());
      if (!inputElemType || !outputElemType)
        return failure();

      const int64_t inputBitWidth = inputElemType.getWidth();
      const int64_t outputBitWidth = outputElemType.getWidth();
      if (outputBitWidth < inputBitWidth)
        return failure();

      Value inputSlice = getOrCreateSlice(op.getInputs().front());
      if (!inputSlice)
        return failure();
      auto extResultType = SliceType::get(ctx, outputBitWidth, vectorLen);
      auto rowNumToExt = rewriter.create<arith::ConstantIntOp>(
          loc, outputBitWidth - inputBitWidth, 64);
      Value newResult = rewriter.create<ExtensionIOp>(loc, extResultType,
                                                      inputSlice, rowNumToExt);

      sliceCache[op] = newResult;
      auto assembledType = inferTensorTypeFromBitsValue(newResult, ctx);
      if (!assembledType)
        return failure();
      Value result = rewriter.create<AssembleOp>(loc, assembledType, newResult);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (op.getInputs().size() != 2 || yieldedOp->getNumOperands() != 2)
      return failure();

    auto inputType =
        dyn_cast<RankedTensorType>(op.getInputs().front().getType());
    if (!inputType || inputType.getRank() != 1)
      return failure();
    auto elemType = inputType.getElementType();
    int64_t bitWidth = elemType.getIntOrFloatBitWidth();
    auto sliceType = SliceType::get(ctx, bitWidth, vectorLen);

    auto lhsArg = dyn_cast<BlockArgument>(yieldedOp->getOperand(0));
    auto rhsArg = dyn_cast<BlockArgument>(yieldedOp->getOperand(1));
    if (!lhsArg || !rhsArg)
      return failure();

    if (lhsArg.getOwner() != &block || rhsArg.getOwner() != &block)
      return failure();

    if (lhsArg.getArgNumber() >= op.getInputs().size() ||
        rhsArg.getArgNumber() >= op.getInputs().size())
      return failure();

    Value lhsInput = op.getInputs()[lhsArg.getArgNumber()];
    Value rhsInput = op.getInputs()[rhsArg.getArgNumber()];

    Value lhsSlice;
    Value rhsSlice;
    if (isa<arith::MulIOp>(yieldedOp)) {
      lhsSlice = getMulOperandSlice(lhsInput);
      rhsSlice = getMulOperandSlice(rhsInput);
    } else {
      lhsSlice = getOrCreateSlice(lhsInput);
      rhsSlice = getOrCreateSlice(rhsInput);
    }
    if (!lhsSlice || !rhsSlice)
      return failure();

    Value newResult;
    if (isa<arith::AddIOp>(yieldedOp)) {
      newResult = rewriter.create<AddIOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::MulIOp>(yieldedOp)) {
      auto lhsSliceType = cast<SliceType>(lhsSlice.getType());
      auto rhsSliceType = cast<SliceType>(rhsSlice.getType());
      auto mulResultSliceType = SliceType::get(
          ctx, lhsSliceType.getBitWidth() + rhsSliceType.getBitWidth(),
          vectorLen);
      newResult =
          rewriter.create<MulIOp>(loc, mulResultSliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::MulFOp>(yieldedOp)) {
      newResult = rewriter.create<MulFOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::AndIOp>(yieldedOp)) {
      newResult = rewriter.create<AndOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::OrIOp>(yieldedOp)) {
      newResult = rewriter.create<OrOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::XOrIOp>(yieldedOp)) {
      newResult = rewriter.create<XOrOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::MaxUIOp>(yieldedOp)) {
      newResult = rewriter.create<MaxOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else if (isa<arith::MinUIOp>(yieldedOp)) {
      newResult = rewriter.create<MinOp>(loc, sliceType, lhsSlice, rhsSlice);
    } else {
      return failure();
    }

    sliceCache[op] = newResult;

    RankedTensorType assembledType = outputType;
    if (!isa<arith::MulFOp>(yieldedOp)) {
      assembledType = inferTensorTypeFromBitsValue(newResult, ctx);
      if (!assembledType)
        return failure();
    }
    Value result = rewriter.create<AssembleOp>(loc, assembledType, newResult);

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct ConvertLinalgFillToBits : public OpConversionPattern<linalg::FillOp> {
  using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, linalg::FillOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOutputs().size() != 1)
      return failure();
    rewriter.replaceOp(op, op.getOutputs().front());
    return success();
  }
};

struct ConvertLinalgMatvecToBits
    : public OpConversionPattern<linalg::MatvecOp> {
  using OpConversionPattern<linalg::MatvecOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatvecOp op, linalg::MatvecOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Value lhsTensor = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value rhsTensor = getSourceTensorBeforeExt(op.getInputs()[1]);

    auto lhsType = dyn_cast<RankedTensorType>(lhsTensor.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsTensor.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 1 || resultType.getRank() != 1)
      return failure();

    auto resultElem = dyn_cast<IntegerType>(resultType.getElementType());
    auto lhsElem = dyn_cast<IntegerType>(lhsType.getElementType());
    auto rhsElem = dyn_cast<IntegerType>(rhsType.getElementType());
    if (!resultElem || !lhsElem || !rhsElem)
      return failure();

    const int64_t m = lhsType.getShape()[0];
    const int64_t k = lhsType.getShape()[1];
    if (rhsType.getShape()[0] != k || resultType.getShape()[0] != m)
      return failure();

    const int64_t lhsBitWidth = lhsElem.getWidth();
    const int64_t rhsBitWidth = rhsElem.getWidth();
    const int64_t resultBitWidth = resultElem.getWidth();
    if (lhsBitWidth + rhsBitWidth != resultBitWidth)
      return failure();

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto emptyBroadcast = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{m, k}, rhsType.getElementType());
    auto rhsBroadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, rhsTensor, emptyBroadcast.getResult(),
        rewriter.getDenseI64ArrayAttr({0}));
    Value rhsBroadcast = rhsBroadcastOp->getResult(0);

    auto resultSliceType = SliceType::get(ctx, resultBitWidth, m);
    auto lhsRowSliceType = RankedTensorType::get({m}, lhsType.getElementType());
    auto rhsRowSliceType = RankedTensorType::get({m}, rhsType.getElementType());
    auto lhsBitsSliceType = SliceType::get(ctx, lhsBitWidth, m);
    auto rhsBitsSliceType = SliceType::get(ctx, rhsBitWidth, m);

    Value resultBitWidthConst = rewriter.create<arith::ConstantIntOp>(
        loc, resultBitWidth, 64);
    Value mConst = rewriter.create<arith::ConstantIntOp>(loc, m, 64);
    Value initAcc = rewriter.create<CreateSliceOp>(loc, resultSliceType,
                                                   resultBitWidthConst, mConst);

    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ub = getOrCreateIndexConstant(rewriter, loc, k);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, initAcc);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value iv = forOp.getInductionVar();
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0), iv};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(m),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};

      Value lhsColumn = rewriter.create<tensor::ExtractSliceOp>(
          loc, lhsRowSliceType, lhsTensor, offsets, sizes, strides);
      Value rhsColumn = rewriter.create<tensor::ExtractSliceOp>(
          loc, rhsRowSliceType, rhsBroadcast, offsets, sizes, strides);

      Value lhsSlice =
          rewriter.create<TransposeOp>(loc, lhsBitsSliceType, lhsColumn);
      Value rhsSlice =
          rewriter.create<TransposeOp>(loc, rhsBitsSliceType, rhsColumn);
      Value mulResult =
          rewriter.create<MulIOp>(loc, resultSliceType, lhsSlice, rhsSlice);
      Value acc = forOp.getRegionIterArgs().front();
      Value nextAcc =
          rewriter.create<AddIOp>(loc, resultSliceType, acc, mulResult);
      rewriter.create<scf::YieldOp>(loc, nextAcc);
    }

    auto assembledType = inferTensorTypeFromBitsValue(forOp.getResult(0), ctx);
    if (!assembledType || assembledType != resultType)
      return failure();
    Value assembled =
        rewriter.create<AssembleOp>(loc, assembledType, forOp.getResult(0));
    rewriter.replaceOp(op, assembled);
    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

struct ConvertLinalgMatmulToBits
    : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, linalg::MatmulOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Value lhsTensor = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value rhsTensor = getSourceTensorBeforeExt(op.getInputs()[1]);
    auto lhsType = dyn_cast<RankedTensorType>(lhsTensor.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsTensor.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 ||
        rhsType.getRank() != 2 || resultType.getRank() != 2)
      return failure();

    auto lhsElem = dyn_cast<IntegerType>(lhsType.getElementType());
    auto rhsElem = dyn_cast<IntegerType>(rhsType.getElementType());
    auto resultElem = dyn_cast<IntegerType>(resultType.getElementType());
    if (!lhsElem || !rhsElem || !resultElem)
      return failure();

    const int64_t m = lhsType.getShape()[0];
    const int64_t k = lhsType.getShape()[1];
    const int64_t n = rhsType.getShape()[1];
    if (rhsType.getShape()[0] != k || resultType.getShape()[0] != m ||
        resultType.getShape()[1] != n)
      return failure();

    if (lhsElem.getWidth() + rhsElem.getWidth() != resultElem.getWidth())
      return failure();

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto colSliceType = SliceType::get(ctx, resultElem.getWidth(), m);
    auto rhsColumnType = RankedTensorType::get({k}, rhsType.getElementType());

    Value colsTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{n}, colSliceType);

    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ub = getOrCreateIndexConstant(rewriter, loc, n);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, colsTensor);

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      Value iv = forOp.getInductionVar();
      SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0), iv};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(k),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
      Value rhsColumn = rewriter.create<tensor::ExtractSliceOp>(
          loc, rhsColumnType, rhsTensor, offsets, sizes, strides);

      Value initVec = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{m}, resultType.getElementType());
      auto matvec = rewriter.create<linalg::MatvecOp>(
          loc, ValueRange{lhsTensor, rhsColumn}, ValueRange{initVec});
      Value matvecSlice =
          rewriter.create<TransposeOp>(loc, colSliceType, matvec->getResult(0));
      Value accTensor = forOp.getRegionIterArgs().front();
      Value updatedTensor =
          rewriter.create<tensor::InsertOp>(loc, matvecSlice, accTensor,
                                            ValueRange{iv});
      rewriter.create<scf::YieldOp>(loc, updatedTensor);
    }

    Value assembled =
        rewriter.create<AssembleOp>(loc, resultType, forOp.getResult(0));
    rewriter.replaceOp(op, assembled);

    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

// Lanes per bank row in the mapper's model (kBankRowWidth in OptimiseMapping).
// An elementwise tensor is cut into tiles of this many lanes so that each tile
// is exactly one full-row bit-serial wave.
static constexpr int64_t kElementwiseTileLanes = 8192;

// Elementwise linalg named ops (linalg.add / linalg.mul / ...) -> a tiled
// scf.for whose body is a single bits slice op.
//
// Why a loop rather than one big slice: bits.transpose maps a whole tensor
// into ONE slice, which lands on ONE bank, so an untiled elementwise op is a
// single row-op on a single bank -- no bank parallelism, and (worse) the trace
// emitter only walks scf.for bodies, so an untiled op emits an EMPTY trace.
// Cutting the tensor into kElementwiseTileLanes-wide tiles produces exactly
// the shape the existing emitter already handles (a parallel scf.for with
// bits.add_i / bits.mul_i directly in its body), so nothing downstream of this
// pattern needs to change.
//
// The tensor is reshaped to <T x L> and the loop walks T, mirroring how
// ConvertLinalgMatmulToBits walks the columns of its rhs.
// True when `v` is an all-zero tensor, either a splat constant or a
// linalg.fill of a zero scalar. Used to recognise ReLU written as
// linalg.max(x, 0).
static bool isZeroTensor(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
    return dense && dense.isSplat() &&
           dense.getSplatValue<APInt>().isZero();
  }
  if (auto fill = dyn_cast<linalg::FillOp>(def)) {
    if (fill.getInputs().size() != 1)
      return false;
    auto cst = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return false;
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    return intAttr && intAttr.getValue().isZero();
  }
  return false;
}

struct ConvertLinalgElementwiseToBits : public ConversionPattern {
  ConvertLinalgElementwiseToBits(MLIRContext *ctx, StringRef opName)
      : ConversionPattern(opName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dpsOp || dpsOp.getDpsInputs().size() != 2 ||
        dpsOp.getDpsInits().size() != 1 || op->getNumResults() != 1)
      return failure();

    Value lhs = dpsOp.getDpsInputs()[0];
    Value rhs = dpsOp.getDpsInputs()[1];
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsTy || !rhsTy || !resTy || lhsTy != rhsTy || lhsTy != resTy)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(lhsTy.getElementType());
    if (!elemTy)
      return failure();

    // linalg.max is only lowered in its ReLU form, max(x, 0). A general
    // bit-serial max needs a comparator + select whose row-op expansion is not
    // implemented in the player, and guessing at its cost would break the
    // exact compiler/gem5 op-count agreement the comparison rests on.
    const bool isMax = op->getName().getStringRef() == "linalg.max";
    bool zeroIsRhs = false;
    if (isMax) {
      if (isZeroTensor(rhs)) {
        zeroIsRhs = true;
      } else if (isZeroTensor(lhs)) {
        zeroIsRhs = false;
      } else {
        return rewriter.notifyMatchFailure(
            op, "only the ReLU form max(x, 0) is supported");
      }
    }

    int64_t numElems = 1;
    for (int64_t d : lhsTy.getShape()) {
      if (d <= 0)
        return failure();
      numElems *= d;
    }
    const int64_t lanes = kElementwiseTileLanes;
    if (numElems % lanes != 0)
      return rewriter.notifyMatchFailure(
          op, "element count must be a multiple of the bank row width");
    const int64_t tiles = numElems / lanes;
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tiles)))
      return rewriter.notifyMatchFailure(
          op, "tile count (elements / bank row width) must be a power of two, "
              "as the mapper requires power-of-two loop trip counts");

    const int64_t bitWidth = elemTy.getWidth();
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Flatten to 1-D. Tile t takes every T-th element starting at t, i.e. the
    // tiles are interleaved rather than contiguous. That is a free choice for
    // an elementwise op (no cross-element dependency) and it makes the layout
    // agree with bits.assemble, which lays a <T x slice<bw x L>> out as
    // <L x T> with the lane index first -- so the assembled tensor collapses
    // straight back to the original 1-D order with no transpose.
    auto flat1dTy = RankedTensorType::get({numElems}, elemTy);
    auto toFlat = [&](Value v) -> Value {
      auto vTy = cast<RankedTensorType>(v.getType());
      if (vTy.getRank() == 1)
        return v;
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < vTy.getRank(); ++i)
        re[0].push_back(i);
      return rewriter.create<tensor::CollapseShapeOp>(loc, flat1dTy, v, re);
    };
    Value lhsFlat = toFlat(lhs);
    Value rhsFlat = toFlat(rhs);

    auto tileSliceTy = SliceType::get(ctx, bitWidth, lanes);
    auto tileTensorTy = RankedTensorType::get({lanes}, elemTy);
    Value acc = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{tiles},
                                                  tileSliceTy);

    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ub = getOrCreateIndexConstant(rewriter, loc, tiles);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, acc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      SmallVector<OpFoldResult> offsets{iv};
      SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(lanes)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(tiles)};
      Value dataFlat = isMax ? (zeroIsRhs ? lhsFlat : rhsFlat) : lhsFlat;
      Value lhsTile = rewriter.create<tensor::ExtractSliceOp>(
          loc, tileTensorTy, dataFlat, offsets, sizes, strides);
      Value lhsSlice =
          rewriter.create<TransposeOp>(loc, tileSliceTy, lhsTile);
      Value rhsSlice;
      if (!isMax) {
        Value rhsTile = rewriter.create<tensor::ExtractSliceOp>(
            loc, tileTensorTy, rhsFlat, offsets, sizes, strides);
        rhsSlice = rewriter.create<TransposeOp>(loc, tileSliceTy, rhsTile);
      }

      Value computed;
      StringRef name = op->getName().getStringRef();
      if (isMax) {
        // Bit-serial ReLU needs no comparator: for two's complement,
        //   ReLU(x) = x AND ~sign(x)
        // where sign(x) is the top bit row. Invert that one row, present it as
        // a full-height slice (a free layout view -- the same DRAM row is just
        // used as an operand once per bit), and AND it into every row.
        // bits.masked_zero tells the scheduler to price this as 4n + 2, which
        // is exactly what gem5's expandRelu emits.
        auto signIdx =
            rewriter.create<arith::ConstantIntOp>(loc, bitWidth - 1, 64);
        auto rowTy = BitRowType::get(ctx, lanes);
        Value signRow =
            rewriter.create<ExtractRowOp>(loc, rowTy, lhsSlice, signIdx);
        Value maskRow = rewriter.create<NotOp>(loc, rowTy, signRow);
        auto widthConst =
            rewriter.create<arith::ConstantIntOp>(loc, bitWidth, 64);
        Value maskSlice = rewriter.create<BroadcastRowOp>(loc, tileSliceTy,
                                                          maskRow, widthConst);
        auto andOp =
            rewriter.create<AndOp>(loc, tileSliceTy, lhsSlice, maskSlice);
        andOp->setAttr("bits.masked_zero", rewriter.getUnitAttr());
        computed = andOp;
      } else if (name == "linalg.add") {
        computed = rewriter.create<AddIOp>(loc, tileSliceTy, lhsSlice, rhsSlice);
      } else if (name == "linalg.mul") {
        // bits.mul_i yields the full 2n-bit product; the named op's result is
        // n bits, so the high half is dropped. The FULL multiply is still
        // charged -- a truncating bit-serial multiply would be cheaper, so
        // this errs on the side of over-reporting our own runtime.
        auto wideTy = SliceType::get(ctx, 2 * bitWidth, lanes);
        Value wide =
            rewriter.create<MulIOp>(loc, wideTy, lhsSlice, rhsSlice);
        auto lowBits = rewriter.create<arith::ConstantIntOp>(loc, bitWidth, 64);
        computed = rewriter.create<ExtractSubSliceOp>(loc, tileSliceTy, wide,
                                                      lowBits);
      } else {
        return failure();
      }
      Value updated = rewriter.create<tensor::InsertOp>(
          loc, computed, forOp.getRegionIterArgs().front(), ValueRange{iv});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    // bits.assemble lays the T slices out as <L x T> (lane index first).
    auto assembledTy = RankedTensorType::get({lanes, tiles}, elemTy);
    Value assembled =
        rewriter.create<AssembleOp>(loc, assembledTy, forOp.getResult(0));
    SmallVector<ReassociationIndices> backRe = {{0, 1}};
    Value flat1d = rewriter.create<tensor::CollapseShapeOp>(loc, flat1dTy,
                                                            assembled, backRe);
    Value result = flat1d;
    if (resTy.getRank() != 1) {
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < resTy.getRank(); ++i)
        re[0].push_back(i);
      result = rewriter.create<tensor::ExpandShapeOp>(loc, resTy, flat1d, re);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Does the target substrate have a widening add? Set from the pass option of
// the same name; see BitsFrontendPasses.td. It is a file-static because the two
// tree builders that consult it are the only readers and both are constructed
// by the same pass instance.
static bool kFrontendWideAdd = true;

// One level of a width-growing adder tree: reduce `cur` (all `width` bits wide)
// pairwise into values one bit wider, returning the new level.
//
// Two spellings of the same arithmetic, which is exactly what the widened-add
// ablation measures:
//   kFrontendWideAdd  : one `bits.add_i_full`, k + k -> k+1, 8k + 2 AAP.
//   !kFrontendWideAdd : zero-extend both operands to k+1 (free -- a reserved
//                       constant row) then an equal-width `bits.add_i`,
//                       8(k+1) + 1 AAP.
// The equal-width form is also the only one a truncating add can express
// losslessly: gem5's expandAddSimdram expands min(lhs_bw, rhs_bw) bits while the
// scheduler prices 8 * result_bw + 1, so the two only agree when
// lhs == rhs == result.
static SmallVector<Value> buildAdderTreeLevel(ConversionPatternRewriter &rewriter,
                                              Location loc,
                                              ArrayRef<Value> cur, int64_t width,
                                              int64_t lanes, Value oneConst) {
  auto wideTy = SliceType::get(rewriter.getContext(), width + 1, lanes);
  SmallVector<Value> next;
  next.reserve((cur.size() + 1) / 2);
  size_t i = 0;
  for (; i + 1 < cur.size(); i += 2) {
    if (kFrontendWideAdd) {
      next.push_back(
          rewriter.create<AddIFullOp>(loc, wideTy, cur[i], cur[i + 1]));
    } else {
      Value l = rewriter.create<ExtensionIOp>(loc, wideTy, cur[i], oneConst);
      Value r = rewriter.create<ExtensionIOp>(loc, wideTy, cur[i + 1], oneConst);
      next.push_back(rewriter.create<AddIOp>(loc, wideTy, l, r));
    }
  }
  if (i < cur.size()) // odd survivor just widens, no arithmetic
    next.push_back(
        rewriter.create<ExtensionIOp>(loc, wideTy, cur[i], oneConst));
  return next;
}

// What a popcount-shaped linalg.generic body counts. `isXnor` distinguishes
// the two kernels this pattern serves:
//   plain popcount : ctpop(arg0)
//   BNN dot product: ctpop(xnor(arg0, arg1))
struct PopcountBodyMatch {
  math::CtPopOp ctpop;
  bool isXnor = false;
  explicit operator bool() const { return ctpop != nullptr; }
};

static bool isAllOnesConstant(Value v) {
  APInt cst;
  if (!matchPattern(v, m_ConstantInt(&cst)))
    return false;
  return cst.isAllOnes();
}

// Strip an optional arith.trunci and return the math.ctpop feeding a
// linalg.generic's yield, together with whether the counted value is a plain
// block argument or an XNOR of the two inputs.
//
// MLIR has no xnor op, so every binary-network kernel in the wild spells it as
// a double xori -- `(a ^ b) ^ -1`. That is the form matched here; recognising
// it is what lets one pattern serve both popcount and BNN, since after the
// XNOR the two kernels are bit-for-bit the same adder tree.
static PopcountBodyMatch matchPopcountBody(linalg::GenericOp op) {
  auto &block = op.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return {};
  Value v = yield.getValues().front();
  if (auto trunc = v.getDefiningOp<arith::TruncIOp>())
    v = trunc.getIn();
  auto ctpop = v.getDefiningOp<math::CtPopOp>();
  if (!ctpop)
    return {};

  auto isArg = [&block](Value val, unsigned n) {
    auto arg = dyn_cast<BlockArgument>(val);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == n;
  };

  PopcountBodyMatch m;
  Value counted = ctpop.getOperand();

  if (isArg(counted, 0)) {
    m.ctpop = ctpop;
    return m;
  }

  auto outer = counted.getDefiningOp<arith::XOrIOp>();
  if (!outer)
    return {};
  Value inner = outer.getLhs(), ones = outer.getRhs();
  if (!isAllOnesConstant(ones)) {
    std::swap(inner, ones);
    if (!isAllOnesConstant(ones))
      return {};
  }
  auto x = inner.getDefiningOp<arith::XOrIOp>();
  if (!x)
    return {};
  if (!((isArg(x.getLhs(), 0) && isArg(x.getRhs(), 1)) ||
        (isArg(x.getLhs(), 1) && isArg(x.getRhs(), 0))))
    return {};

  m.ctpop = ctpop;
  m.isXnor = true;
  return m;
}

// linalg.generic { math.ctpop } -> a tiled scf.for whose body is a fully
// unrolled width-growing adder tree.
//
// Popcount is the kernel bit-serial PuD is built for: the data is already
// transposed, so counting set bits is just adds over bit ROWS -- no per-word
// instruction, no shifts.
//
// Why a tree and why it is unrolled: a linear accumulator (acc += x[i], B
// times at the accumulator's full width) costs B*(8w+1) -- for B=64,w=7 that
// is 3648 AAP versus 1527 for the tree, i.e. 2.4x worse. Since this kernel
// exists to SHOW the PuD advantage, running it 2.4x slower than necessary
// would argue against our own case. The tree cannot be a loop over levels
// (each level has a different bit width, and a loop body has one static
// shape), but it can be straight-line code inside the tile loop: the trace
// emitter reads getComputeLatency() and the operand widths per op, so a body
// holding 63 differently-sized bits.add_i flattens correctly with no change
// downstream.
//
// Why every add is equal-width: gem5's expandAddSimdram expands
// min(lhs_bw, rhs_bw) bits while the scheduler prices 8*result_bw + 1, so the
// two only agree when lhs == rhs == result. Both operands are therefore
// zero-extended to k+1 bits BEFORE each add, turning every tree level into an
// equal-width add that both sides already agree on exactly. bits.ext_i is
// excluded from the mapper and costs nothing -- physically it is the reserved
// constant-zero row used as an operand.
//
// Residual idealisation: a true widening add needs only 8k + 2 AAP (k full
// adders + carry init + carry-out store) against the 8(k+1) + 1 charged here,
// so the tree is over-priced by ~41% (1527 vs 1086 AAP for B=64). That errs
// towards reporting OUR OWN runtime as slower, which is the safe direction;
// closing it needs a widening-add record kind on both sides.
struct ConvertLinalgPopcountToBits
    : public OpConversionPattern<linalg::GenericOp> {
  ConvertLinalgPopcountToBits(MLIRContext *ctx)
      : OpConversionPattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PopcountBodyMatch body = matchPopcountBody(op);
    if (!body)
      return failure();
    const size_t wantInputs = body.isXnor ? 2 : 1;
    if (op.getInputs().size() != wantInputs || op.getOutputs().size() != 1 ||
        op->getNumResults() != 1)
      return failure();
    if (body.isXnor && op.getInputs()[0].getType() != op.getInputs()[1].getType())
      return rewriter.notifyMatchFailure(
          op, "both XNOR operands must have the same tensor type");
    for (utils::IteratorType it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    for (AffineMap map : op.getIndexingMapsArray())
      if (!map.isIdentity())
        return failure();

    auto inTy = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !resTy || inTy.getShape() != resTy.getShape())
      return failure();
    auto inElem = dyn_cast<IntegerType>(inTy.getElementType());
    auto outElem = dyn_cast<IntegerType>(resTy.getElementType());
    if (!inElem || !outElem)
      return failure();

    const int64_t srcBits = inElem.getWidth();
    const int64_t outBits = outElem.getWidth();
    // Widest possible count is srcBits, needing floor(log2(srcBits)) + 1 bits.
    const int64_t treeBits = (int64_t)llvm::Log2_64(srcBits) + 1;
    if (outBits < treeBits)
      return rewriter.notifyMatchFailure(
          op, "result element type is too narrow to hold the population count");

    int64_t numElems = 1;
    for (int64_t d : inTy.getShape()) {
      if (d <= 0)
        return failure();
      numElems *= d;
    }
    const int64_t lanes = kElementwiseTileLanes;
    if (numElems % lanes != 0)
      return rewriter.notifyMatchFailure(
          op, "element count must be a multiple of the bank row width");
    const int64_t tiles = numElems / lanes;
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tiles)))
      return rewriter.notifyMatchFailure(
          op, "tile count must be a power of two");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto flat1dTy = RankedTensorType::get({numElems}, inElem);
    auto flatten = [&](Value v) {
      if (inTy.getRank() == 1)
        return v;
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < inTy.getRank(); ++i)
        re[0].push_back(i);
      return Value(
          rewriter.create<tensor::CollapseShapeOp>(loc, flat1dTy, v, re));
    };
    Value src = flatten(op.getInputs()[0]);
    Value src2 = body.isXnor ? flatten(op.getInputs()[1]) : Value();

    auto srcSliceTy = SliceType::get(ctx, srcBits, lanes);
    auto outSliceTy = SliceType::get(ctx, outBits, lanes);
    auto rowTy = BitRowType::get(ctx, lanes);
    auto tileTensorTy = RankedTensorType::get({lanes}, inElem);

    Value acc = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{tiles},
                                                  outSliceTy);
    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ub = getOrCreateIndexConstant(rewriter, loc, tiles);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, acc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      auto extractTile = [&](Value from) {
        Value t = rewriter.create<tensor::ExtractSliceOp>(
            loc, tileTensorTy, from, SmallVector<OpFoldResult>{iv},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(lanes)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(tiles)});
        return Value(rewriter.create<TransposeOp>(loc, srcSliceTy, t));
      };
      Value x = extractTile(src);
      // The BNN multiply. With +/-1 encoded in one bit, an XNOR IS the
      // multiply, and everything below -- the whole adder tree -- is the
      // dot product's reduction. That is the entire kernel: one 7-command
      // -per-bit row op replaces a full bit-serial integer multiply.
      if (body.isXnor)
        x = rewriter.create<XNOrOp>(loc, srcSliceTy, x, extractTile(src2));

      // Level 0: each bit row becomes a 1-bit slice. Both ops are layout
      // views and cost nothing.
      SmallVector<Value> cur;
      cur.reserve(srcBits);
      auto oneConst = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
      auto oneBitTy = SliceType::get(ctx, 1, lanes);
      for (int64_t i = 0; i < srcBits; ++i) {
        auto idx = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
        Value row = rewriter.create<ExtractRowOp>(loc, rowTy, x, idx);
        cur.push_back(
            rewriter.create<BroadcastRowOp>(loc, oneBitTy, row, oneConst));
      }

      // Width-growing tree: each node is a WIDENING add, k + k -> k+1, which
      // keeps the carry-out row for 8k + 2 AAP. Pre-extending both operands
      // and using a truncating bits.add_i instead would cost 8(k+1) + 1 --
      // 1527 AAP against 1086 for a 64-input tree. That fallback is what
      // `wide-add=false` selects; see buildAdderTreeLevel.
      int64_t width = 1;
      while (cur.size() > 1) {
        cur = buildAdderTreeLevel(rewriter, loc, cur, width, lanes, oneConst);
        ++width;
      }

      Value count = cur.front();
      if (width < outBits) {
        auto pad =
            rewriter.create<arith::ConstantIntOp>(loc, outBits - width, 64);
        count = rewriter.create<ExtensionIOp>(loc, outSliceTy, count, pad);
      }
      Value updated = rewriter.create<tensor::InsertOp>(
          loc, count, forOp.getRegionIterArgs().front(), ValueRange{iv});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    auto assembledTy = RankedTensorType::get({lanes, tiles}, outElem);
    Value assembled =
        rewriter.create<AssembleOp>(loc, assembledTy, forOp.getResult(0));
    SmallVector<ReassociationIndices> backRe = {{0, 1}};
    Value result = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({numElems}, outElem), assembled, backRe);
    if (resTy.getRank() != 1) {
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < resTy.getRank(); ++i)
        re[0].push_back(i);
      result = rewriter.create<tensor::ExpandShapeOp>(loc, resTy, result, re);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// linalg.generic { arith.maxui | arith.minui } reducing the trailing dimension
// -> a tile loop over the parallel dimension wrapping a sequential reduction.
//
// This is the selection half of a k-nearest-neighbour search: with the LANE
// dimension carrying the queries and the reduction dimension walking the
// database, every lane maintains its own running best independently. That is
// the whole reason for this layout -- putting the database in the lanes would
// make top-k a cross-lane horizontal reduction, the one shape a PuD substrate
// is bad at (it is why the popcount and BitWeaving counts are left to the
// host).
//
// k=1 only. General top-k is the same machinery with k carried registers and a
// compare-exchange chain per element, which needs a multi-result match.
// Squared-L2 distance reduction:  out[i][j] = sum_k (a[i][k] - b[j][k])^2.
//
// Matched as a linalg.generic with matmul-shaped maps -- (d0,d2), (d1,d2) ->
// (d0,d1) with iterators [parallel, parallel, reduction] -- whose body is a
// difference, a widening square and an accumulate. That is the same loop
// structure ConvertLinalgMatmulToBits lowers (an outer walk over the second
// parallel dim, an inner reduction with a slice accumulator), with a different
// three-op body, so the skeleton below deliberately mirrors matmul/matvec: the
// second input is broadcast across the lane dimension once, then each reduction
// step extracts one column of each operand, transposes it into a bit slice and
// applies the body.
//
// Widths follow the IR exactly rather than being inferred: the kernel extends
// both i8 operands before subtracting (an i8 difference does not fit in i8) and
// squares into twice that width, so the emitted ops are
//   ext_i(a) , ext_i(b)  -> the difference width  (free: a reserved row)
//   sub_i                -> difference width
//   mul_i                -> 2x difference width  (the widening square)
//   add_i                -> accumulator width
// The extensions are free in the cost model but not fictitious: they are the
// reserved zero/sign row used as an operand, exactly as in the popcount tree.
//
// Restricted to ONE reduction dimension and to the (x-y)^2 body on purpose. A
// general "any elementwise body reduced over a dimension" pattern is the right
// eventual shape, but each body op admitted has to have a priced bits op behind
// it, and inventing costs for ops the simulator cannot expand is precisely the
// drift this pipeline is built to avoid.
struct ConvertLinalgSquaredDistanceToBits
    : public OpConversionPattern<linalg::GenericOp> {
  ConvertLinalgSquaredDistanceToBits(MLIRContext *ctx)
      : OpConversionPattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  // Peel arith.extsi/extui chains, returning the ultimate source value.
  static Value stripExt(Value v) {
    while (auto *def = v.getDefiningOp()) {
      if (isa<arith::ExtSIOp, arith::ExtUIOp>(def))
        v = def->getOperand(0);
      else
        break;
    }
    return v;
  }

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1 ||
        op->getNumResults() != 1)
      return failure();

    auto iters = op.getIteratorTypesArray();
    if (iters.size() != 3 || iters[0] != utils::IteratorType::parallel ||
        iters[1] != utils::IteratorType::parallel ||
        iters[2] != utils::IteratorType::reduction)
      return failure();

    // Maps must be exactly the matmul ones.
    MLIRContext *ctx = rewriter.getContext();
    auto d = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 ||
        maps[0] != AffineMap::get(3, 0, {d(0), d(2)}, ctx) ||
        maps[1] != AffineMap::get(3, 0, {d(1), d(2)}, ctx) ||
        maps[2] != AffineMap::get(3, 0, {d(0), d(1)}, ctx))
      return failure();

    // Body: yield(add(acc, mul(sub(ext(a), ext(b)), same))).
    Block &block = op.getRegion().front();
    if (block.getNumArguments() != 3)
      return failure();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();
    auto add = yield.getValues().front().getDefiningOp<arith::AddIOp>();
    if (!add)
      return failure();
    Value accArg = block.getArgument(2);
    Value squared;
    if (add.getLhs() == accArg)
      squared = add.getRhs();
    else if (add.getRhs() == accArg)
      squared = add.getLhs();
    else
      return failure();
    auto mul = squared.getDefiningOp<arith::MulIOp>();
    if (!mul)
      return failure();
    // A square: both multiplicands trace back to the same difference.
    Value ml = stripExt(mul.getLhs()), mr = stripExt(mul.getRhs());
    if (ml != mr)
      return rewriter.notifyMatchFailure(op, "multiply is not a square");
    auto sub = ml.getDefiningOp<arith::SubIOp>();
    if (!sub)
      return failure();
    if (stripExt(sub.getLhs()) != block.getArgument(0) ||
        stripExt(sub.getRhs()) != block.getArgument(1))
      return rewriter.notifyMatchFailure(
          op, "difference operands are not the two inputs, in order");

    Value aTensor = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value bTensor = getSourceTensorBeforeExt(op.getInputs()[1]);
    auto aTy = dyn_cast<RankedTensorType>(aTensor.getType());
    auto bTy = dyn_cast<RankedTensorType>(bTensor.getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!aTy || !bTy || !resTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
        resTy.getRank() != 2)
      return failure();
    auto aElem = dyn_cast<IntegerType>(aTy.getElementType());
    auto bElem = dyn_cast<IntegerType>(bTy.getElementType());
    auto resElem = dyn_cast<IntegerType>(resTy.getElementType());
    if (!aElem || !bElem || !resElem || aElem != bElem)
      return failure();

    const int64_t lanes = aTy.getShape()[0];   // d0, the SIMD lane dimension
    const int64_t dbs = bTy.getShape()[0];     // d1
    const int64_t feats = aTy.getShape()[1];   // d2, the reduction
    if (bTy.getShape()[1] != feats || resTy.getShape()[0] != lanes ||
        resTy.getShape()[1] != dbs)
      return failure();

    // Widths taken from the body: the subtraction's width and the square's.
    auto diffElem = dyn_cast<IntegerType>(sub.getType());
    auto sqElem = dyn_cast<IntegerType>(mul.getType());
    if (!diffElem || !sqElem)
      return failure();
    const int64_t inW = aElem.getWidth();
    const int64_t diffW = diffElem.getWidth();
    const int64_t sqW = sqElem.getWidth();
    const int64_t accW = resElem.getWidth();
    if (diffW < inW || sqW != 2 * diffW || accW < sqW)
      return rewriter.notifyMatchFailure(
          op, "expected widths in<=diff, square == 2*diff, acc >= square");

    Location loc = op.getLoc();
    auto accSliceTy = SliceType::get(ctx, accW, lanes);
    auto colSliceTy = SliceType::get(ctx, accW, lanes);
    auto inColTensorTy = RankedTensorType::get({lanes}, aTy.getElementType());
    auto inSliceTy = SliceType::get(ctx, inW, lanes);
    auto diffSliceTy = SliceType::get(ctx, diffW, lanes);
    auto sqSliceTy = SliceType::get(ctx, sqW, lanes);

    Value extRows = rewriter.create<arith::ConstantIntOp>(loc, diffW - inW, 64);
    Value accWConst = rewriter.create<arith::ConstantIntOp>(loc, accW, 64);
    Value lanesConst = rewriter.create<arith::ConstantIntOp>(loc, lanes, 64);

    // Broadcast the database across the lane dimension once, so that a
    // reduction step can extract a lanes-long column of it -- the same trick
    // ConvertLinalgMatvecToBits uses for its vector operand.
    auto bEmpty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{lanes, dbs, feats}, bTy.getElementType());
    auto bBroadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, bTensor, bEmpty.getResult(), rewriter.getDenseI64ArrayAttr({0}));
    Value bBroadcast = bBroadcastOp->getResult(0);

    Value colsTensor = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{dbs}, colSliceTy);
    Value zero = getOrCreateIndexConstant(rewriter, loc, 0);
    Value one = getOrCreateIndexConstant(rewriter, loc, 1);
    Value dbsUb = getOrCreateIndexConstant(rewriter, loc, dbs);
    Value featsUb = getOrCreateIndexConstant(rewriter, loc, feats);

    auto outerFor =
        rewriter.create<scf::ForOp>(loc, zero, dbsUb, one, colsTensor);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());
      Value j = outerFor.getInductionVar();

      Value initAcc = rewriter.create<CreateSliceOp>(loc, accSliceTy, accWConst,
                                                     lanesConst);
      auto innerFor =
          rewriter.create<scf::ForOp>(loc, zero, featsUb, one, initAcc);
      {
        OpBuilder::InsertionGuard guard2(rewriter);
        rewriter.setInsertionPointToStart(innerFor.getBody());
        Value k = innerFor.getInductionVar();

        SmallVector<OpFoldResult> aOffsets{rewriter.getIndexAttr(0), k};
        SmallVector<OpFoldResult> aSizes{rewriter.getIndexAttr(lanes),
                                         rewriter.getIndexAttr(1)};
        SmallVector<OpFoldResult> ones2{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
        Value aCol = rewriter.create<tensor::ExtractSliceOp>(
            loc, inColTensorTy, aTensor, aOffsets, aSizes, ones2);

        SmallVector<OpFoldResult> bOffsets{rewriter.getIndexAttr(0), j, k};
        SmallVector<OpFoldResult> bSizes{rewriter.getIndexAttr(lanes),
                                         rewriter.getIndexAttr(1),
                                         rewriter.getIndexAttr(1)};
        SmallVector<OpFoldResult> ones3{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};
        Value bCol = rewriter.create<tensor::ExtractSliceOp>(
            loc, inColTensorTy, bBroadcast, bOffsets, bSizes, ones3);

        Value aSlice = rewriter.create<TransposeOp>(loc, inSliceTy, aCol);
        Value bSlice = rewriter.create<TransposeOp>(loc, inSliceTy, bCol);
        Value aWide = aSlice, bWide = bSlice;
        if (diffW != inW) {
          aWide = rewriter.create<ExtensionIOp>(loc, diffSliceTy, aSlice,
                                               extRows);
          bWide = rewriter.create<ExtensionIOp>(loc, diffSliceTy, bSlice,
                                               extRows);
        }
        Value diff =
            rewriter.create<SubIOp>(loc, diffSliceTy, aWide, bWide);
        Value sq = rewriter.create<MulIOp>(loc, sqSliceTy, diff, diff);
        Value sqAcc = sq;
        if (accW != sqW)
          sqAcc = rewriter.create<ExtensionIOp>(
              loc, accSliceTy, sq,
              rewriter.create<arith::ConstantIntOp>(loc, accW - sqW, 64));
        Value acc = innerFor.getRegionIterArgs().front();
        Value next = rewriter.create<AddIOp>(loc, accSliceTy, acc, sqAcc);
        rewriter.create<scf::YieldOp>(loc, next);
      }

      Value accTensor = outerFor.getRegionIterArgs().front();
      Value updated = rewriter.create<tensor::InsertOp>(
          loc, innerFor.getResult(0), accTensor, ValueRange{j});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    Value assembled =
        rewriter.create<AssembleOp>(loc, resTy, outerFor.getResult(0));
    rewriter.replaceOp(op, assembled);
    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

// Binarised dot product over an UNPACKED i1 tensor:
//   out[i][j] = sum_k  xnor(w[i][k], a[j][k])       (counted as 0/1)
//
// Why a second BNN lowering exists. The original benchmark/bnn kernel packs 64
// binary values into an i64 and reduces them with math.ctpop, which caps the
// reduction at one 64-wide tile -- ctpop has no multi-word form, so a real
// 4096-deep layer cannot be spelled that way at all. This pattern takes the
// natural i1 spelling instead: a [parallel, parallel, reduction] generic with
// matmul-shaped maps, i.e. the same skeleton ConvertLinalgSquaredDistanceToBits
// matches, whose body is the standard double-xori XNOR, a zero-extension and an
// accumulate. The reduction dimension is then free to be as deep as the layer.
//
// **The reduction is emitted as a width-growing adder tree, not as the linear
// accumulator chain the linalg body literally describes.** That is the whole
// point of the pattern and it is not an optimisation flourish: taking the body
// at face value would charge the accumulator's full width on every one of the
// K steps -- K*(8*accW + 1), i.e. 1,052,672 AAP at K=4096/accW=32 -- against
// 73,614 AAP for the tree, 14x worse. The tree is legal here for exactly the
// reason it is legal in the popcount lowering: the summands are 1-bit and the
// addition is associative, so any bracketing yields the same count. Each node
// is a WIDENING add (k + k -> k+1, `bits.add_i_full`, 8k + 2 AAP), so level w
// costs (K / 2^w) * (8w + 2) and the total is independent of how the tree is
// bracketed -- a fact worth knowing when comparing against a hand-written
// word-at-a-time BNN, which lands on the identical row-op count.
//
// The K leaves are materialised straight-line inside the per-output loop rather
// than through an inner scf.for, because a tree needs all its leaves live at
// once; a loop body has one static shape and could only express the chain. That
// makes this the largest body in the suite (K xnors + K-1 adds), which is fine
// downstream: `aggregateLoopWork` in the mapper prices any body over
// kMaxIlpNodesPerLoopBody in closed form instead of lifting it into the ILP.
struct ConvertLinalgBinaryDotToBits
    : public OpConversionPattern<linalg::GenericOp> {
  ConvertLinalgBinaryDotToBits(MLIRContext *ctx)
      : OpConversionPattern<linalg::GenericOp>(ctx, /*benefit=*/3) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1 ||
        op->getNumResults() != 1)
      return failure();

    auto iters = op.getIteratorTypesArray();
    if (iters.size() != 3 || iters[0] != utils::IteratorType::parallel ||
        iters[1] != utils::IteratorType::parallel ||
        iters[2] != utils::IteratorType::reduction)
      return failure();

    MLIRContext *ctx = rewriter.getContext();
    auto d = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 ||
        maps[0] != AffineMap::get(3, 0, {d(0), d(2)}, ctx) ||
        maps[1] != AffineMap::get(3, 0, {d(1), d(2)}, ctx) ||
        maps[2] != AffineMap::get(3, 0, {d(0), d(1)}, ctx))
      return failure();

    // Body: yield(addi(acc, extui(xori(xori(a0, a1), true)))).
    Block &block = op.getRegion().front();
    if (block.getNumArguments() != 3)
      return failure();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();
    auto add = yield.getValues().front().getDefiningOp<arith::AddIOp>();
    if (!add)
      return failure();
    Value accArg = block.getArgument(2);
    Value contrib;
    if (add.getLhs() == accArg)
      contrib = add.getRhs();
    else if (add.getRhs() == accArg)
      contrib = add.getLhs();
    else
      return failure();
    auto ext = contrib.getDefiningOp<arith::ExtUIOp>();
    if (!ext)
      return rewriter.notifyMatchFailure(
          op, "the XNOR must be zero-extended into the accumulator");
    auto outer = ext.getIn().getDefiningOp<arith::XOrIOp>();
    if (!outer)
      return failure();
    Value inner = outer.getLhs(), ones = outer.getRhs();
    if (!isAllOnesConstant(ones)) {
      std::swap(inner, ones);
      if (!isAllOnesConstant(ones))
        return failure();
    }
    auto x = inner.getDefiningOp<arith::XOrIOp>();
    if (!x)
      return failure();
    auto isArg = [&block](Value v, unsigned n) {
      auto a = dyn_cast<BlockArgument>(v);
      return a && a.getOwner() == &block && a.getArgNumber() == n;
    };
    if (!((isArg(x.getLhs(), 0) && isArg(x.getRhs(), 1)) ||
          (isArg(x.getLhs(), 1) && isArg(x.getRhs(), 0))))
      return rewriter.notifyMatchFailure(
          op, "the XNOR operands are not the two inputs");

    Value wTensor = op.getInputs()[0];
    Value aTensor = op.getInputs()[1];
    auto wTy = dyn_cast<RankedTensorType>(wTensor.getType());
    auto aTy = dyn_cast<RankedTensorType>(aTensor.getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!wTy || !aTy || !resTy || wTy.getRank() != 2 || aTy.getRank() != 2 ||
        resTy.getRank() != 2)
      return failure();
    auto wElem = dyn_cast<IntegerType>(wTy.getElementType());
    auto aElem = dyn_cast<IntegerType>(aTy.getElementType());
    auto resElem = dyn_cast<IntegerType>(resTy.getElementType());
    if (!wElem || !aElem || !resElem || wElem.getWidth() != 1 ||
        aElem.getWidth() != 1)
      return rewriter.notifyMatchFailure(op, "both operands must be i1");

    const int64_t lanes = wTy.getShape()[0]; // d0, the SIMD lane dimension
    const int64_t outs = aTy.getShape()[0];  // d1
    const int64_t k = wTy.getShape()[1];     // d2, the reduction
    if (aTy.getShape()[1] != k || resTy.getShape()[0] != lanes ||
        resTy.getShape()[1] != outs)
      return failure();
    if (k <= 1 || !llvm::isPowerOf2_64(static_cast<uint64_t>(k)))
      return rewriter.notifyMatchFailure(
          op, "the reduction depth must be a power of two greater than one");

    // A K-input tree of 1-bit summands tops out at log2(K) + 1 bits.
    const int64_t treeBits = (int64_t)llvm::Log2_64(k) + 1;
    // Block the K leaves so the reduction has a loop the mapper can distribute
    // over banks. 128 blocks matches the largest bank budget we target; a
    // shallower reduction falls back to whatever splits evenly.
    const int64_t blockSize = std::max<int64_t>(2, k / 128);
    const int64_t numBlocks = k / blockSize;
    const int64_t blockBits = (int64_t)llvm::Log2_64(blockSize) + 1;
    const int64_t chainBits = blockBits + (int64_t)llvm::Log2_64_Ceil(numBlocks);
    const int64_t accW = resElem.getWidth();
    if (accW < treeBits)
      return rewriter.notifyMatchFailure(
          op, "result element type is too narrow to hold the count");

    Location loc = op.getLoc();
    auto oneBitTy = SliceType::get(ctx, 1, lanes);
    auto accSliceTy = SliceType::get(ctx, accW, lanes);
    auto colTensorTy = RankedTensorType::get({lanes}, wTy.getElementType());

    // Broadcast the activations across the lane dimension once, exactly as the
    // matvec / squared-distance lowerings do with their second operand.
    auto aEmpty = rewriter.create<tensor::EmptyOp>(
        loc, ArrayRef<int64_t>{lanes, outs, k}, aTy.getElementType());
    auto aBroadcastOp = rewriter.create<linalg::BroadcastOp>(
        loc, aTensor, aEmpty.getResult(), rewriter.getDenseI64ArrayAttr({0}));
    Value aBroadcast = aBroadcastOp->getResult(0);

    Value colsTensor =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{outs}, accSliceTy);
    Value zero = getOrCreateIndexConstant(rewriter, loc, 0);
    Value one = getOrCreateIndexConstant(rewriter, loc, 1);
    Value outsUb = getOrCreateIndexConstant(rewriter, loc, outs);

    auto outerFor = rewriter.create<scf::ForOp>(loc, zero, outsUb, one, colsTensor);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(outerFor.getBody());
      Value j = outerFor.getInductionVar();
      Value oneConst = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);

      SmallVector<OpFoldResult> ones2{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> ones3{rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1),
                                      rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> wSizes{rewriter.getIndexAttr(lanes),
                                       rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> aSizes{rewriter.getIndexAttr(lanes),
                                       rewriter.getIndexAttr(1),
                                       rewriter.getIndexAttr(1)};

      // Block the reduction. The tree itself has to stay straight-line -- its
      // levels differ in shape, so they cannot be folded into a loop body --
      // but the reduction does not have to be ONE tree. Splitting the K leaves
      // into `numBlocks` blocks of `blockSize`, growing a tree inside each and
      // chaining the blocks through an scf.for reduction, costs almost nothing:
      // a tree's total is fixed by how many nodes sit at each width
      // ($K/2^w$), which does not depend on how the leaves are grouped. What it
      // buys is a reduction loop the mapper can see, and therefore a bank
      // count it can choose -- without it the whole tree lands on one bank and
      // its 4095 additions serialise there.
      Value blocksUb = getOrCreateIndexConstant(rewriter, loc, numBlocks);
      auto blockSliceTy = SliceType::get(ctx, blockBits, lanes);
      auto chainSliceTy = SliceType::get(ctx, chainBits, lanes);
      Value chainWConst =
          rewriter.create<arith::ConstantIntOp>(loc, chainBits, 64);
      Value lanesConst2 =
          rewriter.create<arith::ConstantIntOp>(loc, lanes, 64);
      Value accInit = rewriter.create<CreateSliceOp>(loc, chainSliceTy,
                                                     chainWConst, lanesConst2);
      auto blockFor =
          rewriter.create<scf::ForOp>(loc, zero, blocksUb, one, accInit);
      {
        OpBuilder::InsertionGuard guard2(rewriter);
        rewriter.setInsertionPointToStart(blockFor.getBody());
        Value blk = blockFor.getInductionVar();
        Value blockSizeCst =
            getOrCreateIndexConstant(rewriter, loc, blockSize);
        Value base = rewriter.create<arith::MulIOp>(loc, blk, blockSizeCst);

        // Leaves of this block: one XNOR each. Straight-line, since the tree
        // below needs every leaf live at once.
        SmallVector<Value> cur;
        cur.reserve(blockSize);
        for (int64_t i = 0; i < blockSize; ++i) {
          Value off = getOrCreateIndexConstant(rewriter, loc, i);
          Value kIdx = rewriter.create<arith::AddIOp>(loc, base, off);
          Value wCol = rewriter.create<tensor::ExtractSliceOp>(
              loc, colTensorTy, wTensor,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), kIdx}, wSizes,
              ones2);
          Value aCol = rewriter.create<tensor::ExtractSliceOp>(
              loc, colTensorTy, aBroadcast,
              SmallVector<OpFoldResult>{rewriter.getIndexAttr(0), j, kIdx},
              aSizes, ones3);
          Value wSlice = rewriter.create<TransposeOp>(loc, oneBitTy, wCol);
          Value aSlice = rewriter.create<TransposeOp>(loc, oneBitTy, aCol);
          cur.push_back(rewriter.create<XNOrOp>(loc, oneBitTy, wSlice, aSlice));
        }

        // Width-growing tree within the block, same shape as the popcount
        // lowering's.
        int64_t width = 1;
        while (cur.size() > 1) {
          cur = buildAdderTreeLevel(rewriter, loc, cur, width, lanes, oneConst);
          ++width;
        }
        Value partial = cur.front();
        if (width < blockBits)
          partial = rewriter.create<ExtensionIOp>(
              loc, blockSliceTy, partial,
              rewriter.create<arith::ConstantIntOp>(loc, blockBits - width, 64));

        // Chain the blocks. This add is the reduction combiner the mapper
        // distributes over banks; it is carried at chainBits so the sum of all
        // numBlocks partial counts is exact.
        Value wide = rewriter.create<ExtensionIOp>(
            loc, chainSliceTy, partial,
            rewriter.create<arith::ConstantIntOp>(loc, chainBits - blockBits,
                                                  64));
        Value next = rewriter.create<AddIOp>(
            loc, chainSliceTy, blockFor.getRegionIterArgs().front(), wide);
        rewriter.create<scf::YieldOp>(loc, next);
      }

      Value count = blockFor.getResult(0);
      if (chainBits < accW)
        count = rewriter.create<ExtensionIOp>(
            loc, accSliceTy, count,
            rewriter.create<arith::ConstantIntOp>(loc, accW - chainBits, 64));
      Value updated = rewriter.create<tensor::InsertOp>(
          loc, count, outerFor.getRegionIterArgs().front(), ValueRange{j});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    Value assembled =
        rewriter.create<AssembleOp>(loc, resTy, outerFor.getResult(0));
    rewriter.replaceOp(op, assembled);
    return success();
  }
};

struct ConvertLinalgMinMaxReduceToBits
    : public OpConversionPattern<linalg::GenericOp> {
  ConvertLinalgMinMaxReduceToBits(MLIRContext *ctx)
      : OpConversionPattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1 ||
        op->getNumResults() != 1)
      return failure();

    auto &block = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();
    Operation *comb = yield.getValues().front().getDefiningOp();
    if (!comb || !isa<arith::MaxUIOp, arith::MinUIOp>(comb))
      return failure();
    const bool isMax = isa<arith::MaxUIOp>(comb);
    // Both operands must be the two block arguments, in either order.
    auto argNo = [&block](Value v) -> int {
      auto a = dyn_cast<BlockArgument>(v);
      if (!a || a.getOwner() != &block)
        return -1;
      return (int)a.getArgNumber();
    };
    int p = argNo(comb->getOperand(0)), q = argNo(comb->getOperand(1));
    if (!((p == 0 && q == 1) || (p == 1 && q == 0)))
      return failure();

    auto iters = op.getIteratorTypesArray();
    if (iters.size() != 2 || iters[0] != utils::IteratorType::parallel ||
        iters[1] != utils::IteratorType::reduction)
      return rewriter.notifyMatchFailure(
          op, "expected one parallel then one reduction iterator");

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[0].isIdentity() ||
        maps[1] != AffineMap::get(2, 0, {rewriter.getAffineDimExpr(0)},
                                  rewriter.getContext()))
      return rewriter.notifyMatchFailure(
          op, "expected (d0,d1)->(d0,d1) and (d0,d1)->(d0) indexing");

    auto inTy = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !resTy || inTy.getRank() != 2 || resTy.getRank() != 1)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(inTy.getElementType());
    if (!elemTy || inTy.getElementType() != resTy.getElementType())
      return failure();

    const int64_t rows = inTy.getShape()[0];   // queries
    const int64_t cols = inTy.getShape()[1];   // database vectors
    if (rows <= 0 || cols <= 0 || resTy.getShape()[0] != rows)
      return failure();

    const int64_t lanes = kElementwiseTileLanes;
    if (rows % lanes != 0)
      return rewriter.notifyMatchFailure(
          op, "parallel extent must be a multiple of the bank row width");
    const int64_t tiles = rows / lanes;
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tiles)))
      return rewriter.notifyMatchFailure(op, "tile count must be a power of two");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Value src = op.getInputs().front();
    auto sliceTy = SliceType::get(ctx, elemTy.getWidth(), lanes);
    auto tileTensorTy = RankedTensorType::get({lanes}, elemTy);

    Value acc = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{tiles},
                                                 sliceTy);
    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ubT = getOrCreateIndexConstant(rewriter, loc, tiles);
    Value ubC = getOrCreateIndexConstant(rewriter, loc, cols);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);

    auto tileLoop = rewriter.create<scf::ForOp>(loc, lb, ubT, step, acc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tileLoop.getBody());
      Value t = tileLoop.getInductionVar();

      // Identity for the running best: 0 for unsigned max, all-ones for min.
      auto widthConst = rewriter.create<arith::ConstantIntOp>(
          loc, elemTy.getWidth(), 64);
      auto lanesConst = rewriter.create<arith::ConstantIntOp>(loc, lanes, 64);
      Value seed =
          isMax ? rewriter.create<CreateSliceOp>(loc, sliceTy, widthConst,
                                                 lanesConst)
                      .getResult()
                : rewriter.create<CreateAllOneSliceOp>(loc, sliceTy, widthConst,
                                                       lanesConst)
                      .getResult();

      auto dbLoop = rewriter.create<scf::ForOp>(loc, lb, ubC, step, seed);
      {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPointToStart(dbLoop.getBody());
        Value j = dbLoop.getInductionVar();
        // Lane l of tile t is query (t + l*tiles): the same interleaving the
        // other tiled kernels use, because bits.assemble lays a
        // tensor<T x slice<bw x L>> out as <L x T>.
        Value col = rewriter.create<tensor::ExtractSliceOp>(
            loc, tileTensorTy, src, SmallVector<OpFoldResult>{t, j},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(lanes),
                                      rewriter.getIndexAttr(1)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(tiles),
                                      rewriter.getIndexAttr(1)});
        Value s = rewriter.create<TransposeOp>(loc, sliceTy, col);
        Value cur = dbLoop.getRegionIterArgs().front();
        Value best =
            isMax ? rewriter.create<MaxOp>(loc, sliceTy, cur, s).getResult()
                  : rewriter.create<MinOp>(loc, sliceTy, cur, s).getResult();
        rewriter.create<scf::YieldOp>(loc, best);
      }

      Value updated = rewriter.create<tensor::InsertOp>(
          loc, dbLoop.getResult(0), tileLoop.getRegionIterArgs().front(),
          ValueRange{t});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    auto assembledTy = RankedTensorType::get({lanes, tiles}, elemTy);
    Value assembled =
        rewriter.create<AssembleOp>(loc, assembledTy, tileLoop.getResult(0));
    SmallVector<ReassociationIndices> backRe = {{0, 1}};
    Value result = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({rows}, elemTy), assembled, backRe);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// One side of a BETWEEN predicate: a comparison of block argument 0 against a
// constant, normalised so the block argument is conceptually on the left.
struct ScanBound {
  bool valid = false;
  bool isLower = false; // v > c / v >= c  (as opposed to v < c / v <= c)
  int64_t constant = 0;
};

// Mirror a predicate about its operands: `c < v` means the same as `v > c`.
static arith::CmpIPredicate swapCmpOperands(arith::CmpIPredicate p) {
  switch (p) {
  case arith::CmpIPredicate::sgt: return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge: return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::slt: return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle: return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::ugt: return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge: return arith::CmpIPredicate::ule;
  case arith::CmpIPredicate::ult: return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule: return arith::CmpIPredicate::uge;
  default: return p;
  }
}

static ScanBound matchScanBound(Value v, Block &block) {
  ScanBound b;
  auto cmp = v.getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return b;
  auto isArg0 = [&block](Value val) {
    auto a = dyn_cast<BlockArgument>(val);
    return a && a.getOwner() == &block && a.getArgNumber() == 0;
  };

  arith::CmpIPredicate pred = cmp.getPredicate();
  APInt cst;
  if (isArg0(cmp.getLhs()) && matchPattern(cmp.getRhs(), m_ConstantInt(&cst))) {
    // already `v <pred> c`
  } else if (isArg0(cmp.getRhs()) &&
             matchPattern(cmp.getLhs(), m_ConstantInt(&cst))) {
    pred = swapCmpOperands(pred);
  } else {
    return b;
  }

  switch (pred) {
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::ugt:
  case arith::CmpIPredicate::uge:
    b.isLower = true;
    break;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    b.isLower = false;
    break;
  default:
    return b; // eq/ne are not range bounds
  }
  b.constant = cst.getSExtValue();
  b.valid = true;
  return b;
}

// linalg.generic { cmpi > c1 ; cmpi < c2 ; andi } -> a tiled scf.for whose
// body is a single bits.range_scan.
//
// This is BitWeaving/V: a database column stored bit-sliced, one bit plane per
// row, so a BETWEEN predicate over N values is N-way SIMD by construction.
// bits.transpose of the column tensor already produces exactly that layout,
// which is why the whole scan collapses to one op.
//
// Strict and inclusive bounds cost the same -- only the value the comparison
// chains are seeded with differs -- so both are accepted and the distinction
// is carried in the attributes rather than the cost.
//
// The COUNT(*) that a real query wraps around this is deliberately left to the
// host: counting a mask across lanes is a horizontal reduction, awkward on PuD
// and done on the CPU by every BitWeaving implementation this models.
struct ConvertLinalgRangeScanToBits
    : public OpConversionPattern<linalg::GenericOp> {
  ConvertLinalgRangeScanToBits(MLIRContext *ctx)
      : OpConversionPattern<linalg::GenericOp>(ctx, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1 ||
        op->getNumResults() != 1)
      return failure();

    auto &block = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();
    auto conj = yield.getValues().front().getDefiningOp<arith::AndIOp>();
    if (!conj)
      return failure();

    ScanBound a = matchScanBound(conj.getLhs(), block);
    ScanBound b = matchScanBound(conj.getRhs(), block);
    if (!a.valid || !b.valid || a.isLower == b.isLower)
      return rewriter.notifyMatchFailure(
          op, "body is not a lower-bound AND upper-bound range predicate");
    const ScanBound &lo = a.isLower ? a : b;
    const ScanBound &hi = a.isLower ? b : a;

    for (utils::IteratorType it : op.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return failure();
    for (AffineMap map : op.getIndexingMapsArray())
      if (!map.isIdentity())
        return failure();

    auto inTy = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !resTy || inTy.getShape() != resTy.getShape())
      return failure();
    auto inElem = dyn_cast<IntegerType>(inTy.getElementType());
    auto outElem = dyn_cast<IntegerType>(resTy.getElementType());
    if (!inElem || !outElem || outElem.getWidth() != 1)
      return rewriter.notifyMatchFailure(
          op, "range scan must produce a one-bit match mask");

    int64_t numElems = 1;
    for (int64_t d : inTy.getShape()) {
      if (d <= 0)
        return failure();
      numElems *= d;
    }
    const int64_t lanes = kElementwiseTileLanes;
    if (numElems % lanes != 0)
      return rewriter.notifyMatchFailure(
          op, "column length must be a multiple of the bank row width");
    const int64_t tiles = numElems / lanes;
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tiles)))
      return rewriter.notifyMatchFailure(op, "tile count must be a power of two");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    auto flat1dTy = RankedTensorType::get({numElems}, inElem);
    Value src = op.getInputs().front();
    if (inTy.getRank() != 1) {
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < inTy.getRank(); ++i)
        re[0].push_back(i);
      src = rewriter.create<tensor::CollapseShapeOp>(loc, flat1dTy, src, re);
    }

    auto colSliceTy = SliceType::get(ctx, inElem.getWidth(), lanes);
    auto maskSliceTy = SliceType::get(ctx, 1, lanes);
    auto tileTensorTy = RankedTensorType::get({lanes}, inElem);

    Value acc = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{tiles},
                                                 maskSliceTy);
    Value lb = getOrCreateIndexConstant(rewriter, loc, 0);
    Value ub = getOrCreateIndexConstant(rewriter, loc, tiles);
    Value step = getOrCreateIndexConstant(rewriter, loc, 1);
    auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, acc);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      Value iv = forOp.getInductionVar();
      Value tile = rewriter.create<tensor::ExtractSliceOp>(
          loc, tileTensorTy, src, SmallVector<OpFoldResult>{iv},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(lanes)},
          SmallVector<OpFoldResult>{rewriter.getIndexAttr(tiles)});
      Value col = rewriter.create<TransposeOp>(loc, colSliceTy, tile);
      Value mask = rewriter.create<RangeScanOp>(
          loc, maskSliceTy, col, rewriter.getI64IntegerAttr(lo.constant),
          rewriter.getI64IntegerAttr(hi.constant));
      Value updated = rewriter.create<tensor::InsertOp>(
          loc, mask, forOp.getRegionIterArgs().front(), ValueRange{iv});
      rewriter.create<scf::YieldOp>(loc, updated);
    }

    auto assembledTy = RankedTensorType::get({lanes, tiles}, outElem);
    Value assembled =
        rewriter.create<AssembleOp>(loc, assembledTy, forOp.getResult(0));
    SmallVector<ReassociationIndices> backRe = {{0, 1}};
    Value result = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get({numElems}, outElem), assembled, backRe);
    if (resTy.getRank() != 1) {
      SmallVector<ReassociationIndices> re(1);
      for (int64_t i = 0; i < resTy.getRank(); ++i)
        re[0].push_back(i);
      result = rewriter.create<tensor::ExpandShapeOp>(loc, resTy, result, re);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Which convolution dimension becomes the bit-serial SIMD lane dimension,
// i.e. the M of the GEMM the convolution lowers to.
enum class ConvLaneDim {
  OutChannel, // M = F      (outer parallel trip = P*Q)
  Spatial     // M = P*Q    (outer parallel trip = F)
};

// linalg.conv_2d_nchw_fchw -> linalg.matmul (im2col), restricted to 1x1.
//
// Why only 1x1: with R=S=1 and unit stride the im2col "patch matrix" IS the
// input, so the lowering is a pure reshape and duplicates nothing. For R*S > 1
// im2col replicates every input pixel R*S times; emitting that as a free
// reshape would hand Cinnamon a data-movement saving that no part of the cost
// model ever pays for, which is exactly the kind of unmodelled-resource win we
// refuse to take. Such convolutions are rejected rather than silently
// mismodelled. (Independently the mapper needs power-of-two loop trip counts
// and the reduction trip is C*R*S, so the usual 3x3 with power-of-two C is
// unmappable regardless.)
//
// Both lane layouts do identical arithmetic but can differ by >4x once the
// reduction loop's chosen bank count saturates the device, so the caller picks.
struct ConvertLinalgConv2DNchwFchwToBits
    : public OpConversionPattern<linalg::Conv2DNchwFchwOp> {
  ConvertLinalgConv2DNchwFchwToBits(MLIRContext *ctx, ConvLaneDim laneDim)
      : OpConversionPattern<linalg::Conv2DNchwFchwOp>(ctx), laneDim(laneDim) {}

  ConvLaneDim laneDim;

  static bool allOnes(DenseIntElementsAttr attr) {
    if (!attr)
      return true; // absent == default of 1
    return llvm::all_of(attr.getValues<int64_t>(),
                        [](int64_t v) { return v == 1; });
  }

  // Single source of truth for "can we lower this convolution". Returns a
  // human-readable reason when we cannot, so both the pattern (via
  // notifyMatchFailure) and the pass's up-front check (via emitError) report
  // the same thing -- the conversion driver otherwise swallows the reason and
  // leaves the user with a bare "failed to legalize".
  static std::optional<std::string> unsupportedReason(linalg::Conv2DNchwFchwOp op) {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return std::string("expected exactly 2 inputs and 1 output");
    if (!allOnes(op.getStrides()) || !allOnes(op.getDilations()))
      return std::string("only unit stride and unit dilation are supported "
                         "(a strided 1x1 im2col is a gather, not a reshape)");

    auto filTy = dyn_cast<RankedTensorType>(
        getSourceTensorBeforeExt(op.getInputs()[1]).getType());
    auto inTy = dyn_cast<RankedTensorType>(
        getSourceTensorBeforeExt(op.getInputs()[0]).getType());
    if (!inTy || !filTy || inTy.getRank() != 4 || filTy.getRank() != 4)
      return std::string("expected 4-D NCHW input and FCHW filter");
    if (filTy.getShape()[2] != 1 || filTy.getShape()[3] != 1)
      return std::string(
          "only 1x1 convolutions are supported: for R*S > 1 the im2col "
          "expansion duplicates every input pixel R*S times, and modelling "
          "that as a free reshape would understate the cost. (The mapper also "
          "requires power-of-two loop trip counts and the reduction trip is "
          "C*R*S, so a 3x3 with power-of-two C is unmappable regardless.)");
    if (inTy.getShape()[0] != 1)
      return std::string("only batch size 1 is supported");
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(linalg::Conv2DNchwFchwOp op,
                  linalg::Conv2DNchwFchwOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto reason = unsupportedReason(op))
      return rewriter.notifyMatchFailure(op, *reason);

    Value input = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value filter = getSourceTensorBeforeExt(op.getInputs()[1]);
    Value outInit = op.getOutputs()[0];

    auto inTy = dyn_cast<RankedTensorType>(input.getType());
    auto filTy = dyn_cast<RankedTensorType>(filter.getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inTy || !filTy || !outTy || inTy.getRank() != 4 ||
        filTy.getRank() != 4 || outTy.getRank() != 4)
      return failure();

    const int64_t batch = inTy.getShape()[0];
    const int64_t c = inTy.getShape()[1];
    const int64_t h = inTy.getShape()[2];
    const int64_t w = inTy.getShape()[3];
    const int64_t f = filTy.getShape()[0];
    const int64_t p = outTy.getShape()[2];
    const int64_t q = outTy.getShape()[3];

    if (filTy.getShape()[1] != c || outTy.getShape()[0] != batch ||
        outTy.getShape()[1] != f || p != h || q != w)
      return failure();

    auto inElem = dyn_cast<IntegerType>(inTy.getElementType());
    auto filElem = dyn_cast<IntegerType>(filTy.getElementType());
    auto outElem = dyn_cast<IntegerType>(outTy.getElementType());
    if (!inElem || !filElem || !outElem)
      return failure();
    // The matmul lowering requires the product to be exactly representable.
    if (inElem.getWidth() + filElem.getWidth() != outElem.getWidth())
      return rewriter.notifyMatchFailure(
          op, "result element width must equal the sum of the operand widths");

    Location loc = op.getLoc();
    const int64_t pq = p * q;

    // 1x1 im2col is pure reshaping: [1,C,H,W] -> [C,P*Q], [F,C,1,1] -> [F,C].
    SmallVector<ReassociationIndices> collapse2x2 = {{0, 1}, {2, 3}};
    SmallVector<ReassociationIndices> collapseFilter = {{0}, {1, 2, 3}};

    auto input2dTy = RankedTensorType::get({c, pq}, inElem);
    Value input2d = rewriter.create<tensor::CollapseShapeOp>(
        loc, input2dTy, input, collapse2x2);

    auto filter2dTy = RankedTensorType::get({f, c}, filElem);
    Value filter2d = rewriter.create<tensor::CollapseShapeOp>(
        loc, filter2dTy, filter, collapseFilter);

    auto out2dTy = RankedTensorType::get({f, pq}, outElem);
    Value out2d = rewriter.create<tensor::CollapseShapeOp>(loc, out2dTy,
                                                            outInit, collapse2x2);

    Value gemm2d;
    if (laneDim == ConvLaneDim::OutChannel) {
      // M = F, K = C, N = P*Q.
      gemm2d = rewriter
                   .create<linalg::MatmulOp>(loc, ValueRange{filter2d, input2d},
                                             ValueRange{out2d})
                   ->getResult(0);
    } else {
      // M = P*Q, K = C, N = F, computed transposed and flipped back.
      // linalg.transpose is a data-layout change, not compute: it is left in
      // the IR and costs nothing, exactly like the linalg.broadcast the matvec
      // lowering already emits.
      Value aInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{pq, c}, inElem);
      Value a = rewriter
                    .create<linalg::TransposeOp>(loc, input2d, aInit,
                                                 ArrayRef<int64_t>{1, 0})
                    ->getResult(0);
      Value bInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{c, f}, filElem);
      Value b = rewriter
                    .create<linalg::TransposeOp>(loc, filter2d, bInit,
                                                 ArrayRef<int64_t>{1, 0})
                    ->getResult(0);
      Value accInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{pq, f}, outElem);
      Value mmT = rewriter
                      .create<linalg::MatmulOp>(loc, ValueRange{a, b},
                                                ValueRange{accInit})
                      ->getResult(0);
      Value backInit = rewriter.create<tensor::EmptyOp>(
          loc, ArrayRef<int64_t>{f, pq}, outElem);
      gemm2d = rewriter
                   .create<linalg::TransposeOp>(loc, mmT, backInit,
                                                ArrayRef<int64_t>{1, 0})
                   ->getResult(0);
    }

    Value result = rewriter.create<tensor::ExpandShapeOp>(loc, outTy, gemm2d,
                                                          collapse2x2);
    rewriter.replaceOp(op, result);
    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

struct ConvertLinalgGenericTruncToBits
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!outputType || (outputType.getRank() != 1 && outputType.getRank() != 2))
      return failure();

    auto yield =
        dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();

    auto trunc = dyn_cast_or_null<arith::TruncIOp>(
        yield.getValues().front().getDefiningOp());
    if (!trunc)
      return failure();

    auto arg = dyn_cast<BlockArgument>(trunc.getIn());
    if (!arg || arg.getOwner() != &op.getRegion().front() ||
        arg.getArgNumber() != 0)
      return failure();

    auto inputAssemble =
        dyn_cast_or_null<bits::AssembleOp>(op.getInputs()[0].getDefiningOp());
    if (!inputAssemble)
      return failure();

    auto assembledType = inferTensorTypeFromBitsValue(inputAssemble.getInput(),
                                                      rewriter.getContext());
    if (!assembledType)
      return failure();
    Value result = rewriter.create<AssembleOp>(op.getLoc(), assembledType,
                                               inputAssemble.getInput());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct EraseDeadLinalgGeneric : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, linalg::GenericOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertLinalgToBits
    : public ConvertLinalgToBitsBase<ConvertLinalgToBits> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto &ctx = getContext();

    ConvLaneDim laneDim;
    if (convLaneDim == "oc") {
      laneDim = ConvLaneDim::OutChannel;
    } else if (convLaneDim == "spatial") {
      laneDim = ConvLaneDim::Spatial;
    } else {
      func.emitError("convert-linalg-to-bits: conv-lane-dim must be 'oc' or "
                     "'spatial', got '")
          << convLaneDim << "'";
      signalPassFailure();
      return;
    }

    // Report deliberately-unsupported convolutions precisely; the conversion
    // driver would otherwise only say "failed to legalize".
    bool badConv = false;
    func.walk([&](linalg::Conv2DNchwFchwOp conv) {
      if (auto reason =
              ConvertLinalgConv2DNchwFchwToBits::unsupportedReason(conv)) {
        conv.emitError("convert-linalg-to-bits: ") << *reason;
        badConv = true;
      }
    });
    if (badConv) {
      signalPassFailure();
      return;
    }

    kFrontendWideAdd = wideAdd;

    RewritePatternSet patterns(&ctx);
    patterns.add<ConvertLinalgMatmulToBits, ConvertLinalgMatvecToBits,
                 ConvertLinalgFillToBits, ConvertLinalgGenericTruncToBits,
                 ConvertLinalgGenericToBits, EraseDeadLinalgGeneric>(&ctx);
    patterns.add<ConvertLinalgConv2DNchwFchwToBits>(&ctx, laneDim);
    patterns.add<ConvertLinalgElementwiseToBits>(&ctx, "linalg.add");
    patterns.add<ConvertLinalgElementwiseToBits>(&ctx, "linalg.mul");
    patterns.add<ConvertLinalgElementwiseToBits>(&ctx, "linalg.max");
    patterns.add<ConvertLinalgPopcountToBits>(&ctx);
    patterns.add<ConvertLinalgRangeScanToBits>(&ctx);
    patterns.add<ConvertLinalgMinMaxReduceToBits>(&ctx);
    patterns.add<ConvertLinalgSquaredDistanceToBits>(&ctx);
    patterns.add<ConvertLinalgBinaryDotToBits>(&ctx);

    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<BitsDialect>();
    target.addLegalOp<linalg::BroadcastOp>();
    // Data-layout ops emitted by the conv lowering; they carry no compute.
    target.addLegalOp<linalg::TransposeOp>();
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      return isExtSIGeneric(op) || isTruncIGeneric(op);
    });
    target.addDynamicallyLegalOp<linalg::YieldOp>(
        [](linalg::YieldOp) { return true; });
    // target.addIllegalOp<linalg::GenericOp>();
    target.addIllegalDialect<linalg::LinalgDialect>();

    if (applyPartialConversion(func, target, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // removeUnnecessaryAssembles(func);
    removeUnusedExtensionsAndConstants(func);
    rewriteTruncGenericsToAssemble(func);
    removeDeadLinalgGenerics(func);
    removeUnusedTensorEmpties(func);
    removeUnnecessaryAssembles(func);
  }

  static void rewriteTruncGenericsToAssemble(func::FuncOp func) {
    SmallVector<linalg::GenericOp, 8> truncGenerics;
    func.walk([&](linalg::GenericOp genericOp) {
      if (isTruncIGeneric(genericOp))
        truncGenerics.push_back(genericOp);
    });

    OpBuilder builder(func.getContext());
    for (linalg::GenericOp genericOp : truncGenerics) {
      auto inputAssemble = dyn_cast_or_null<bits::AssembleOp>(
          genericOp.getInputs()[0].getDefiningOp());
      if (!inputAssemble)
        continue;

      auto outputType = inferTensorTypeFromBitsValue(inputAssemble.getInput(),
                                                     func.getContext());
      if (!outputType)
        continue;

      builder.setInsertionPoint(genericOp);
      Value newAssemble = builder.create<bits::AssembleOp>(
          genericOp.getLoc(), outputType, inputAssemble.getInput());
      genericOp.getResult(0).replaceAllUsesWith(newAssemble);
      genericOp.erase();
    }
  }

  static void removeDeadLinalgGenerics(func::FuncOp func) {
    SmallVector<Operation *, 8> toErase;
    func.walk([&](linalg::GenericOp genericOp) {
      if (genericOp->use_empty())
        toErase.push_back(genericOp);
    });

    for (Operation *op : toErase)
      op->erase();
  }

  static void removeUnusedTensorEmpties(func::FuncOp func) {
    SmallVector<Operation *, 8> toErase;
    func.walk([&](tensor::EmptyOp emptyOp) {
      if (emptyOp.getResult().use_empty()) {
        toErase.push_back(emptyOp);
      }
    });

    for (Operation *op : toErase)
      op->erase();
  }

  static void removeUnnecessaryAssembles(func::FuncOp func) {
    bool changed = true;
    while (changed) {
      changed = false;

      SmallVector<bits::TransposeOp, 8> transposesToErase;
      SmallVector<bits::AssembleOp, 8> assemblesToErase;

      func.walk([&](bits::TransposeOp transposeOp) {
        auto assembleOp =
            dyn_cast_or_null<bits::AssembleOp>(transposeOp.getInput().getDefiningOp());
        if (!assembleOp)
          return;

        if (transposeOp.getResult().getType() != assembleOp.getInput().getType())
          return;

        transposeOp.getResult().replaceAllUsesWith(assembleOp.getInput());
        transposesToErase.push_back(transposeOp);
        if (assembleOp.getResult().use_empty())
          assemblesToErase.push_back(assembleOp);
        changed = true;
      });

      for (bits::TransposeOp transposeOp : transposesToErase)
        transposeOp.erase();
      for (bits::AssembleOp assembleOp : assemblesToErase)
        if (assembleOp->use_empty())
          assembleOp.erase();
    }

    llvm::SmallPtrSet<Value, 8> returnedValues;
    func.walk([&](func::ReturnOp returnOp) {
      for (Value operand : returnOp.getOperands())
        returnedValues.insert(operand);
    });

    SmallVector<Operation *> toErase;
    func.walk([&](bits::AssembleOp assembleOp) {
      Value result = assembleOp.getResult();

      if (!returnedValues.contains(result) && result.use_empty()) {
        toErase.push_back(assembleOp);
      }
    });

    for (Operation *op : toErase)
      op->erase();
  }

  static void removeUnusedExtensionsAndConstants(func::FuncOp func) {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;

      func.walk([&](bits::ExtensionIOp extOp) {
        if (extOp.getResult().use_empty()) {
          toErase.push_back(extOp);
        }
      });

      func.walk([&](arith::ConstantOp cstOp) {
        if (cstOp.getResult().use_empty()) {
          toErase.push_back(cstOp);
        }
      });

      if (toErase.empty())
        break;

      changed = true;
      for (Operation *op : toErase)
        op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> bits_frontend::createConvertLinalgToBitsPass() {
  return std::make_unique<ConvertLinalgToBits>();
}
