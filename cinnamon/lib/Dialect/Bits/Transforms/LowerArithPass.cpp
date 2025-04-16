#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSLOWERARITHPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bits {
struct LowerAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bits::AddOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = cast<SliceType>(op.getResult().getType());
    auto bitType = BitType::get(rewriter.getContext());
    Value bitWidth = rewriter.create<arith::ConstantIntOp>(loc, resultType.getBitWidth(), rewriter.getI64Type());
    Value vectorLength = rewriter.create<arith::ConstantIntOp>(loc, resultType.getVectorLength(), rewriter.getI64Type());
    Value resultSlice = rewriter.create<CreateSliceOp>(loc, resultType, bitWidth, vectorLength);

    Value lower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value outerUpper = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getVectorLength());
    Value innerUpper = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getBitWidth());

    auto outerLoop = rewriter.create<scf::ForOp>(
        loc, lower, outerUpper, step, ValueRange{resultSlice});

    rewriter.setInsertionPointToStart(outerLoop.getBody());

    Value vecIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), outerLoop.getInductionVar());
    Value slice = outerLoop.getRegionIterArgs()[0];
    Value carry = rewriter.create<CreateBitOp>(loc, bitType);

    auto innerLoop = rewriter.create<scf::ForOp>(
        loc, lower, innerUpper, step, ValueRange{slice, carry});

    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value bitIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), innerLoop.getInductionVar());
    Value accSlice = innerLoop.getRegionIterArgs()[0];
    Value cin = innerLoop.getRegionIterArgs()[1];

    Value lhsBit = rewriter.create<ExtractOp>(loc, bitType, lhs, bitIdx, vecIdx);
    Value rhsBit = rewriter.create<ExtractOp>(loc, bitType, rhs, bitIdx, vecIdx);
    
    auto addBit = rewriter.create<AddBitOp>(loc, TypeRange{bitType, bitType}, lhsBit, rhsBit, cin);
    auto sumBit = addBit.getResult(0);
    auto cout = addBit.getResult(1);

    Value updatedSlice = rewriter.create<InsertOp>(loc, resultType, accSlice, sumBit, bitIdx, vecIdx);

    rewriter.create<scf::YieldOp>(loc, ValueRange{updatedSlice, cout});

    rewriter.setInsertionPointAfter(innerLoop);
    rewriter.create<scf::YieldOp>(loc, innerLoop.getResult(0));

    rewriter.setInsertionPointAfter(outerLoop);

    rewriter.replaceOp(op, outerLoop.getResult(0));
    return success();
  }
};

struct BitsLowerArithPass 
    : public ::impl::BitsLowerArithPassBase<BitsLowerArithPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAddOpPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::bits

std::unique_ptr<mlir::Pass> mlir::bits::createBitsLowerArithPass() {
  return std::make_unique<BitsLowerArithPass>();
}

