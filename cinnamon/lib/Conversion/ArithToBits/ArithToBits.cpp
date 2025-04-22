#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsFrontendPasses.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include <memory>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

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
    auto &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    patterns.add<
        ConvertArithTensorOpToBits<arith::AddIOp, AddOp>>(&ctx);
    
    ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal([](...) { return true; });
    target.addLegalDialect<BitsDialect>();
    target.addIllegalOp<arith::AddIOp>();

    if (applyPartialConversion(getOperation(), target, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> bits_frontend::createConvertArithToBitsPass() {
  return std::make_unique<ConvertArithToBits>();
}
