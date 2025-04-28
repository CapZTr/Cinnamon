#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsFrontendPasses.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include <cstdint>
#include <memory>
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

    auto sliceType = SliceType::get(
      rewriter.getContext(),
      elemType.getIntOrFloatBitWidth(),
      lhsType.getShape()[0]);
    Value lhsSlice = rewriter.create<TransposeOp>(loc, sliceType, lhs);
    Value rhsSlice = rewriter.create<TransposeOp>(loc, sliceType, rhs);
    Value resultSlice = rewriter.create<TargetOp>(loc, sliceType, lhsSlice, rhsSlice);

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

    simplify(func);
  }

  static void simplify(func::FuncOp func) {
    SmallVector<std::tuple<OpOperand *, Value>, 8> toRewire;
    SmallVector<Operation *, 8> toErase;

    func->walk([&](AssembleOp assemble) {
      bool usedByReturn = false;

      for (Operation *user : assemble->getUsers()) {
        if (auto returnOp = dyn_cast<func::ReturnOp>(user)) {
          usedByReturn = true;
          break;
        }
      }

      if (usedByReturn)
        return;

      SmallVector<Operation *, 2> transposesToErase;
      unsigned userCount = 0;
      for (Operation *user : assemble->getUsers()) {
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
