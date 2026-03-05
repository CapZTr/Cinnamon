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

    auto getExtSIGenericSource = [&](Value tensorValue) -> Value {
      auto generic = dyn_cast_or_null<linalg::GenericOp>(
          tensorValue.getDefiningOp());
      if (!generic || generic.getInputs().size() != 1
          || generic.getOutputs().size() != 1) {
        return {};
      }

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
    };

    auto getMulOperandSlice = [&](Value tensorOperand) -> Value {
      if (auto source = getExtSIGenericSource(tensorOperand)) {
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
      Value result = rewriter.create<AssembleOp>(loc, outputType, newResult);
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

    Value lhsSlice = getOrCreateSlice(lhsInput);
    Value rhsSlice = getOrCreateSlice(rhsInput);
    if (isa<arith::MulIOp>(yieldedOp)) {
      lhsSlice = getMulOperandSlice(lhsInput);
      rhsSlice = getMulOperandSlice(rhsInput);
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

    Value result = rewriter.create<AssembleOp>(loc, outputType, newResult);

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct ConvertLinalgToBits
    : public ConvertLinalgToBitsBase<ConvertLinalgToBits> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<ConvertLinalgGenericToBits>(&ctx);
    
    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<BitsDialect>();
    // target.addIllegalOp<linalg::GenericOp>();
    target.addIllegalDialect<linalg::LinalgDialect>();

    if (applyPartialConversion(func, target, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    removeUnnecessaryAssembles(func);
    removeUnusedExtensionsAndConstants(func);
    removeUnusedTensorEmpties(func);
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
