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
    
    if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
      return failure();

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

    auto outputType = dyn_cast<RankedTensorType>(op.getResultTypes().front());
    if (!outputType || outputType.getRank() != 1)
      return failure();

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs().front().getType());
    if (!inputType || inputType.getRank() != 1)
      return failure();

    Location loc = op.getLoc();
    auto elemType = inputType.getElementType();
    auto ctx = rewriter.getContext();
    int64_t bitWidth = elemType.getIntOrFloatBitWidth();
    int64_t vectorLen = inputType.getShape()[0];

    auto &inputCache = InputCache::get();
    auto &sliceCache = SliceCache::get();

    auto sliceType = SliceType::get(ctx, bitWidth, vectorLen);

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

      Value slice = rewriter.create<TransposeOp>(loc, sliceType, operand);

      inputCache[operand] = slice;

      return slice;
    };

    auto &block = op.getRegion().front();
    if (block.getOperations().size() != 2)
      return failure();

    auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
    if (!yieldOp || yieldOp.getValues().size() != 1)
      return failure();

    Value yieldedValue = yieldOp.getValues().front();
    Operation *yieldedOp = yieldedValue.getDefiningOp();
    if (!yieldedOp || yieldedOp->getNumOperands() != 2)
      return failure();

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

    Value newResult;
    if (isa<arith::AddIOp>(yieldedOp)) {
      newResult = rewriter.create<AddIOp>(loc, sliceType, lhsSlice, rhsSlice);
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
    target.addIllegalOp<linalg::GenericOp>();

    if (applyPartialConversion(func, target, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    removeUnnecessaryAssembles(func);
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

  static void simplify(func::FuncOp func) {
    SmallVector<std::tuple<OpOperand *, Value>, 8> toRewire;
    SmallVector<Operation *, 8> toErase;

    func->walk([&](AssembleOp assemble) {
      bool usedByReturn = false;

      for (Operation* user : assemble->getUsers()) {
        if (auto returnOp = dyn_cast<func::ReturnOp>(user)) {
          usedByReturn = true;
          break;
        }
      }

      if (usedByReturn)
        return;

      SmallVector<Operation *, 2> transposesToErase;
      unsigned userCount = 0;
      for (Operation* user : assemble->getUsers()) {
        ++userCount;
        auto transpose = dyn_cast<TransposeOp>(user);
        // if (!transpose || transpose->hasOneUse())
        //   continue;

        auto add = dyn_cast<AddIOp>(*transpose->user_begin());
        // if (!add)
        //   continue;

        Value originalSlice = assemble.getInput();
        for (OpOperand &operand : add->getOpOperands()) {
          if (operand.get() == transpose.getOutput()) {
            toRewire.emplace_back(&operand, originalSlice);
          }
        }

        transposesToErase.push_back(transpose);
      }

      if (transposesToErase.size() == userCount) {
        for (auto *transpose : transposesToErase) {
          toErase.push_back(transpose);
        }
        toErase.push_back(assemble);
      }
    });

    for (auto [operandPtr, slice] : toRewire) {
      operandPtr->set(slice);
    }

    for (auto *op : toErase) {
      op->erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> bits_frontend::createConvertLinalgToBitsPass() {
  return std::make_unique<ConvertLinalgToBits>();
}
