#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsFrontendPasses.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <cstdint>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

template<typename SourceOp, typename TargetOp>
struct ConvertArithTensorOpToBits : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      SourceOp::Adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    Location loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());

    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "Operands must be RankedTensorType");

    if (lhsType.getRank() != 1 || rhsType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "Only 1-D tensors are supported");

    auto elemType = lhsType.getElementType();
    if (!elemType.isSignlessInteger())
      return rewriter.notifyMatchFailure(op, "Tensor elements must be signless integer");

    auto ctx = rewriter.getContext();
    int64_t bitWidth = elemType.getIntOrFloatBitWidth();
    int64_t vectorLen = lhsType.getShape()[0];

    auto &inputCache = InputCache::get();
    auto &sliceCache = SliceCache::get();

    auto getOrCreateSlice = [&](Value operand) -> Value {
      Operation* defOp = operand.getDefiningOp();
      if (defOp) {
        auto it = sliceCache.find(defOp);
        if (it != sliceCache.end())
          return it->second;
      }

      auto it = inputCache.find(operand);
      if (it != inputCache.end())
        return it->second;

      auto sliceType = SliceType::get(ctx, bitWidth, vectorLen);
      Value slice = rewriter.create<TransposeOp>(loc, sliceType, operand);

      inputCache[operand] = slice;

      return slice;
    };

    Value lhsSlice = getOrCreateSlice(lhs);
    Value rhsSlice = getOrCreateSlice(rhs);

    auto sliceType = SliceType::get(ctx, bitWidth, vectorLen);
    Value resultSlice = rewriter.create<TargetOp>(loc, sliceType, lhsSlice, rhsSlice);

    sliceCache[op] = resultSlice;

    Value result = rewriter.create<AssembleOp>(loc, lhsType, resultSlice);

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct ConvertArithToBits
    : public ConvertArithToBitsBase<ConvertArithToBits> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<
        ConvertArithTensorOpToBits<arith::AddIOp, AddOp>>(&ctx);
    
    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<BitsDialect>();
    target.addIllegalOp<arith::AddIOp>();

    if (applyPartialConversion(func, target, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    // simplify(func);
    removeUnnecessaryAssembles(func);
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

        auto add = dyn_cast<AddOp>(*transpose->user_begin());
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

std::unique_ptr<Pass> bits_frontend::createConvertArithToBitsPass() {
  return std::make_unique<ConvertArithToBits>();
}
