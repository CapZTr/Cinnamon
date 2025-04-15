// #include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
// #include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/DialectConversion.h"

// using namespace mlir;
// using namespace bits;

// namespace {
// struct LowerAddOpPattern : OpRewritePattern<bits::AddOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(bits::AddOp op, PatternRewriter &rewriter) const override {
//     auto loc = op.getLoc();
//     auto lhs = op.getLhs();
//     auto rhs = op.getRhs();
//     auto resultType = cast<SliceType>(op.getResult().getType());

//     auto initSlice = rewriter.create<CreateSliceOp>(loc, resultType);

//     auto outerLoop = rewriter.create<scf::ForOp>(
//         loc, rewriter.getIndexAttr(0), rewriter.getIndexAttr(resultType.getBitWidth()),
//         rewriter.getIndexAttr(1), ValueRange{initSlice});

//     rewriter.setInsertionPointToStart(outerLoop.getBody());

//     auto bitIdx = outerLoop.getInductionVar();
//     auto inSlice = outerLoop.getRegionIterArgs()[0];

//     auto innerLoop = rewriter.create<scf::ForOp>(
//         loc, rewriter.getIndexAttr(0), rewriter.getIndexAttr(resultType.getVectorLength()),
//         rewriter.getIndexAttr(1), ValueRange{inSlice});

//     rewriter.setInsertionPointToStart(innerLoop.getBody());

//     auto vecIdx = innerLoop.getInductionVar();
//     auto accSlice = innerLoop.getRegionIterArgs()[0];

//     auto lhsBit = rewriter.create<ExtractOp>(
//         loc, rewriter.getType<mlir::bits::BitType>(), lhs, bitIdx, vecIdx);
//     auto rhsBit = rewriter.create<ExtractOp>(
//         loc, rewriter.getType<mlir::bits::BitType>(), rhs, bitIdx, vecIdx);
    
//     // Computation
//     auto zeroBit = rewriter.create<CreateBitOp>(
//         loc, lhsBit.getType());

//     auto sumBit = rewriter.create<AddBitOp>(
//         loc, lhsBit.getType(), lhsBit, rhsBit, zeroBit).getResult(0);

//     auto updatedSlice = rewriter.create<InsertOp>(
//         loc, accSlice.getType(), accSlice, sumBit, bitIdx, vecIdx)->getResult(0);

//     rewriter.create<mlir::scf::YieldOp>(loc, updatedSlice);

//     rewriter.setInsertionPointAfter(innerLoop);
//     rewriter.create<mlir::scf::YieldOp>(loc, innerLoop.getResult(0));

//     rewriter.setInsertionPointAfter(outerLoop);

//     rewriter.replaceOp(op, outerLoop.getResult(0));
//     return success();
//   }
// };
// } // namespace
