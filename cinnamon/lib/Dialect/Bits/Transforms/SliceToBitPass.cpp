#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

#define GEN_PASS_DEF_BITSSLICETOBITPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bits {
struct LowerAddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bits::AddOp op, PatternRewriter &rewriter) const override {
    // auto loc = op.getLoc();
    // Value lhs = op.getLhs();
    // Value rhs = op.getRhs();
    // auto resultType = cast<SliceType>(op.getResult().getType());
    // auto bitType = BitType::get(rewriter.getContext());
    // Value bitWidth = rewriter.create<arith::ConstantIntOp>(loc, resultType.getBitWidth(), rewriter.getI64Type());
    // Value vectorLength = rewriter.create<arith::ConstantIntOp>(loc, resultType.getVectorLength(), rewriter.getI64Type());
    // Value resultSlice = rewriter.create<CreateSliceOp>(loc, resultType, bitWidth, vectorLength);

    // Value lower = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Value outerUpper = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getVectorLength());
    // Value innerUpper = rewriter.create<arith::ConstantIndexOp>(loc, resultType.getBitWidth());

    // auto outerLoop = rewriter.create<scf::ForOp>(
    //     loc, lower, outerUpper, step, ValueRange{resultSlice});

    // rewriter.setInsertionPointToStart(outerLoop.getBody());

    // Value vecIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), outerLoop.getInductionVar());
    // Value slice = outerLoop.getRegionIterArgs()[0];
    // Value carry = rewriter.create<CreateBitOp>(loc, bitType);

    // auto innerLoop = rewriter.create<scf::ForOp>(
    //     loc, lower, innerUpper, step, ValueRange{slice, carry});

    // rewriter.setInsertionPointToStart(innerLoop.getBody());

    // Value bitIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), innerLoop.getInductionVar());
    // Value accSlice = innerLoop.getRegionIterArgs()[0];
    // Value cin = innerLoop.getRegionIterArgs()[1];

    // Value lhsBit = rewriter.create<ExtractOp>(loc, bitType, lhs, bitIdx, vecIdx);
    // Value rhsBit = rewriter.create<ExtractOp>(loc, bitType, rhs, bitIdx, vecIdx);
    
    // auto addBit = rewriter.create<AddBitOp>(loc, TypeRange{bitType, bitType}, lhsBit, rhsBit, cin);
    // auto sumBit = addBit.getResult(0);
    // auto cout = addBit.getResult(1);

    // Value updatedSlice = rewriter.create<InsertOp>(loc, resultType, accSlice, sumBit, bitIdx, vecIdx);

    // rewriter.create<scf::YieldOp>(loc, ValueRange{updatedSlice, cout});

    // rewriter.setInsertionPointAfter(innerLoop);
    // rewriter.create<scf::YieldOp>(loc, innerLoop.getResult(0));

    // rewriter.setInsertionPointAfter(outerLoop);

    // rewriter.replaceOp(op, outerLoop.getResult(0));

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = cast<SliceType>(op.getResult().getType());

    int64_t bitWidth = resultType.getBitWidth();
    int64_t vectorLength = resultType.getVectorLength();
    auto i64Type = rewriter.getI64Type();
    auto bitType = BitType::get(rewriter.getContext());

    Value resultSlice = rewriter.create<CreateSliceOp>(
        loc, resultType,
        rewriter.create<arith::ConstantIntOp>(loc, bitWidth, i64Type).getResult(),
        rewriter.create<arith::ConstantIntOp>(loc, vectorLength, i64Type).getResult());

    auto outerLoop = rewriter.create<affine::AffineForOp>(loc, 0, vectorLength, 1, ValueRange{resultSlice});
    rewriter.setInsertionPointToStart(outerLoop.getBody());

    Value vecIdx = rewriter.create<arith::IndexCastOp>(loc, i64Type, outerLoop.getInductionVar());
    Value carry = rewriter.create<CreateBitOp>(loc, bitType); // carry = 0
    Value accSlice = outerLoop.getRegionIterArgs()[0];

    auto innerLoop = rewriter.create<affine::AffineForOp>(loc, 0, bitWidth, 1, ValueRange{accSlice, carry});
    rewriter.setInsertionPointToStart(innerLoop.getBody());

    Value bitIdx = rewriter.create<arith::IndexCastOp>(loc, i64Type, innerLoop.getInductionVar());

    Value lhsBit = rewriter.create<ExtractOp>(loc, bitType, lhs, bitIdx, vecIdx);
    Value rhsBit = rewriter.create<ExtractOp>(loc, bitType, rhs, bitIdx, vecIdx);

    auto addBit = rewriter.create<AddBitOp>(loc, TypeRange{bitType, bitType}, lhsBit, rhsBit, carry);
    Value sumBit = addBit.getResult(0);
    Value cout = addBit.getResult(1);

    Value updatedSlice = rewriter.create<InsertOp>(loc, resultType, accSlice, sumBit, bitIdx, vecIdx);

    rewriter.create<affine::AffineYieldOp>(loc, ValueRange{updatedSlice, cout});

    rewriter.setInsertionPointAfter(innerLoop);

    rewriter.create<affine::AffineYieldOp>(loc, innerLoop.getResult(0));

    rewriter.setInsertionPointAfter(outerLoop);

    rewriter.replaceOp(op, outerLoop.getResult(0));

    return success();
  }
};

struct BitsSliceToBitPass 
    : public ::impl::BitsSliceToBitPassBase<BitsSliceToBitPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerAddOpPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::bits

std::unique_ptr<mlir::Pass> mlir::bits::createBitsSliceToBitPass() {
  return std::make_unique<BitsSliceToBitPass>();
}
