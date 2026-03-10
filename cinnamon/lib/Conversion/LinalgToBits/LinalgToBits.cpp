#include "cinm-mlir/Conversion/LinalgToBits/LinalgToBits.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
  if (!generic || generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
    return false;

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto ext = dyn_cast_or_null<arith::ExtSIOp>(yield.getValues().front().getDefiningOp());
  if (!ext)
    return false;

  auto arg = dyn_cast<BlockArgument>(ext.getIn());
  return arg && arg.getOwner() == &gBlock && arg.getArgNumber() == 0;
}

static bool isTruncIGeneric(linalg::GenericOp generic) {
  if (!generic || generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
    return false;

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;

  auto trunc = dyn_cast_or_null<arith::TruncIOp>(yield.getValues().front().getDefiningOp());
  if (!trunc)
    return false;

  auto arg = dyn_cast<BlockArgument>(trunc.getIn());
  return arg && arg.getOwner() == &gBlock && arg.getArgNumber() == 0;
}

static Value stripExtSIGeneric(Value tensorValue) {
  auto generic = dyn_cast_or_null<linalg::GenericOp>(tensorValue.getDefiningOp());
  if (!generic || generic.getInputs().size() != 1 || generic.getOutputs().size() != 1)
    return {};

  auto &gBlock = generic.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(gBlock.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return {};

  auto ext = dyn_cast_or_null<arith::ExtSIOp>(yield.getValues().front().getDefiningOp());
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
  auto generic = dyn_cast_or_null<linalg::GenericOp>(tensorValue.getDefiningOp());
  if (!generic || !isExtSIGeneric(generic) || !generic->use_empty())
    return;
  rewriter.eraseOp(generic);
}

static Value createTransposeFromTensor(Value tensor,
                                       ConversionPatternRewriter &rewriter,
                                       Location loc,
                                       MLIRContext *ctx) {
  auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorType)
    return {};

  auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
  if (!elementType)
    return {};

  if (tensorType.getRank() == 1) {
    auto outputType = SliceType::get(ctx, elementType.getWidth(), tensorType.getShape()[0]);
    return rewriter.create<TransposeOp>(loc, outputType, tensor);
  }

  if (tensorType.getRank() == 2) {
    auto outputType = CubeType::get(
        ctx, elementType.getWidth(), tensorType.getShape()[0], tensorType.getShape()[1]);
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

struct ConvertLinalgGenericToBits
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op,
      linalg::GenericOp::Adaptor,
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

      if (auto assemble =
              dyn_cast_or_null<bits::AssembleOp>(tensorOperand.getDefiningOp())) {
        if (auto ext =
                dyn_cast_or_null<bits::ExtensionIOp>(assemble.getInput().getDefiningOp())) {
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

      auto inputType = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
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
      Value newResult = rewriter.create<ExtensionIOp>(loc,
                                                      extResultType,
                                                      inputSlice,
                                                      rowNumToExt);

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

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
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
      auto mulResultSliceType =
          SliceType::get(ctx,
                         lhsSliceType.getBitWidth() + rhsSliceType.getBitWidth(),
                         vectorLen);
      newResult = rewriter.create<MulIOp>(loc, mulResultSliceType, lhsSlice,
                                          rhsSlice);
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

  LogicalResult matchAndRewrite(
      linalg::FillOp op,
      linalg::FillOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getOutputs().size() != 1)
      return failure();
    rewriter.replaceOp(op, op.getOutputs().front());
    return success();
  }
};

struct ConvertLinalgMatvecToBits : public OpConversionPattern<linalg::MatvecOp> {
  using OpConversionPattern<linalg::MatvecOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::MatvecOp op,
      linalg::MatvecOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Value lhsTensor = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value rhsTensor = getSourceTensorBeforeExt(op.getInputs()[1]);

    auto lhsType = dyn_cast<RankedTensorType>(lhsTensor.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsTensor.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 || rhsType.getRank() != 1 ||
        resultType.getRank() != 1)
      return failure();

    auto lhsElem = dyn_cast<IntegerType>(lhsType.getElementType());
    if (!lhsElem)
      return failure();

    Value lhsCube = createTransposeFromTensor(lhsTensor, rewriter, op.getLoc(), rewriter.getContext());
    Value rhsSlice = createTransposeFromTensor(rhsTensor, rewriter, op.getLoc(), rewriter.getContext());
    if (!lhsCube || !rhsSlice)
      return failure();

    auto matvecResultType = SliceType::get(rewriter.getContext(), lhsElem.getWidth() * 2, lhsType.getShape()[0]);
    Value matvec = rewriter.create<MatvecIOp>(op.getLoc(), matvecResultType, lhsCube, rhsSlice);
    auto assembledType = inferTensorTypeFromBitsValue(matvec, rewriter.getContext());
    if (!assembledType)
      return failure();
    Value assembled = rewriter.create<AssembleOp>(op.getLoc(), assembledType, matvec);
    rewriter.replaceOp(op, assembled);
    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

struct ConvertLinalgMatmulToBits : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::MatmulOp op,
      linalg::MatmulOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

    Value lhsTensor = getSourceTensorBeforeExt(op.getInputs()[0]);
    Value rhsTensor = getSourceTensorBeforeExt(op.getInputs()[1]);
    auto lhsType = dyn_cast<RankedTensorType>(lhsTensor.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhsTensor.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        resultType.getRank() != 2)
      return failure();

    auto lhsElem = dyn_cast<IntegerType>(lhsType.getElementType());
    if (!lhsElem)
      return failure();

    Value lhsCube = createTransposeFromTensor(lhsTensor, rewriter, op.getLoc(), rewriter.getContext());
    Value rhsCube = createTransposeFromTensor(rhsTensor, rewriter, op.getLoc(), rewriter.getContext());
    if (!lhsCube || !rhsCube)
      return failure();

    auto matmulResultType = CubeType::get(
        rewriter.getContext(), lhsElem.getWidth() * 2, lhsType.getShape()[0], rhsType.getShape()[1]);
    Value matmul = rewriter.create<MatmulIOp>(op.getLoc(), matmulResultType, lhsCube, rhsCube);
    auto assembledType = inferTensorTypeFromBitsValue(matmul, rewriter.getContext());
    if (!assembledType)
      return failure();
    Value assembled = rewriter.create<AssembleOp>(op.getLoc(), assembledType, matmul);

    auto canBypassTruncUsers = [&]() {
      if (assembledType == resultType)
        return true;
      if (op->use_empty())
        return true;

      for (Operation *user : op->getUsers()) {
        auto genericUser = dyn_cast<linalg::GenericOp>(user);
        if (!genericUser || !isTruncIGeneric(genericUser))
          return false;
        auto genericResultType =
            dyn_cast<RankedTensorType>(genericUser.getResult(0).getType());
        if (!genericResultType || genericResultType != assembledType)
          return false;
      }
      return true;
    };

    if (!canBypassTruncUsers())
      return failure();

    if (assembledType == resultType) {
      rewriter.replaceOp(op, assembled);
    } else {
      SmallVector<Operation *, 4> truncUsers;
      for (Operation *user : op->getUsers())
        truncUsers.push_back(user);

      for (Operation *user : truncUsers) {
        auto genericUser = cast<linalg::GenericOp>(user);
        genericUser.getResult(0).replaceAllUsesWith(assembled);
        rewriter.eraseOp(genericUser);
      }
      rewriter.eraseOp(op);
    }

    eraseExtSIGenericIfDead(op.getInputs()[0], rewriter);
    eraseExtSIGenericIfDead(op.getInputs()[1], rewriter);
    return success();
  }
};

struct ConvertLinalgGenericTruncToBits
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op,
      linalg::GenericOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    auto outputType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!outputType || (outputType.getRank() != 1 && outputType.getRank() != 2))
      return failure();

    auto yield = dyn_cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
    if (!yield || yield.getValues().size() != 1)
      return failure();

    auto trunc = dyn_cast_or_null<arith::TruncIOp>(yield.getValues().front().getDefiningOp());
    if (!trunc)
      return failure();

    auto arg = dyn_cast<BlockArgument>(trunc.getIn());
    if (!arg || arg.getOwner() != &op.getRegion().front() || arg.getArgNumber() != 0)
      return failure();

    auto inputAssemble = dyn_cast_or_null<bits::AssembleOp>(op.getInputs()[0].getDefiningOp());
    if (!inputAssemble)
      return failure();

    auto assembledType =
        inferTensorTypeFromBitsValue(inputAssemble.getInput(), rewriter.getContext());
    if (!assembledType)
      return failure();
    Value result = rewriter.create<AssembleOp>(op.getLoc(), assembledType, inputAssemble.getInput());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct EraseDeadLinalgGeneric : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp op,
      linalg::GenericOp::Adaptor,
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

    RewritePatternSet patterns(&ctx);
    patterns.add<ConvertLinalgMatmulToBits, ConvertLinalgMatvecToBits,
                 ConvertLinalgFillToBits, ConvertLinalgGenericTruncToBits,
                 ConvertLinalgGenericToBits, EraseDeadLinalgGeneric>(&ctx);
    
    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<BitsDialect>();
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      return isExtSIGeneric(op) || isTruncIGeneric(op);
    });
    target.addDynamicallyLegalOp<linalg::YieldOp>([](linalg::YieldOp) {
      return true;
    });
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
      auto inputAssemble =
          dyn_cast_or_null<bits::AssembleOp>(genericOp.getInputs()[0].getDefiningOp());
      if (!inputAssemble)
        continue;

      auto outputType = inferTensorTypeFromBitsValue(inputAssemble.getInput(), func.getContext());
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

    for (Operation* op : toErase)
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
