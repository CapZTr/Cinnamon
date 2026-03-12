#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>


using namespace mlir;

namespace mlir::bits {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSCUBETOSLICEPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

struct LowerMatvecIPattern : public OpRewritePattern<MatvecIOp> {
  using OpRewritePattern<MatvecIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatvecIOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value cube = op.getLhs();
    Value slice = op.getRhs();

    auto cubeType = dyn_cast<CubeType>(cube.getType());
    auto sliceType = dyn_cast<SliceType>(slice.getType());
    if (!cubeType || !sliceType)
      return failure();

    auto bitwidth = cubeType.getBitWidth();
    auto m = cubeType.getVectorLength();
    auto k = cubeType.getHeight();

    auto layerSliceType = SliceType::get(ctx, bitwidth, m);

    Value resBitwidthVal = rewriter.create<arith::ConstantIntOp>(loc, bitwidth * 2, 64);
    Value mVal = rewriter.create<arith::ConstantIntOp>(loc, m, 64);

    auto resSliceType = SliceType::get(ctx, bitwidth * 2, m);
    Value resSlice = rewriter.create<CreateSliceOp>(loc, resSliceType, resBitwidthVal, mVal);

    Value c0Val = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value kIndexVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto loop = rewriter.create<scf::ForOp>(loc, c0Val, kIndexVal, c1Val,
        ValueRange{resSlice},
        [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs){
          Value curRes = iterArgs[0];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value mulLhs = builder.create<ExtractSliceOp>(bodyLoc, layerSliceType, cube, ivI64);
          Value mulRhs = builder.create<ExtractAndReplicateColumnOp>(bodyLoc, layerSliceType, slice, ivI64, mVal);
          Value product = builder.create<bits::MulIOp>(bodyLoc, resSliceType, mulLhs, mulRhs);

          Value accSlice = builder.create<bits::AddIOp>(bodyLoc, resSliceType, curRes, product);

          builder.create<scf::YieldOp>(bodyLoc, accSlice);
        });
    
    rewriter.replaceOp(op, loop.getResults());

    return success();
  }
};

struct LowerMatmulIPattern : public OpRewritePattern<MatmulIOp> {
  using OpRewritePattern<MatmulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulIOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value lhsCube = op.getLhs();
    Value rhsCube = op.getRhs();

    auto lhsCubeType = dyn_cast<CubeType>(lhsCube.getType());
    auto rhsCubeType = dyn_cast<CubeType>(rhsCube.getType());
    if (!lhsCubeType || !rhsCubeType)
      return failure();

    auto bitwidth = lhsCubeType.getBitWidth();
    auto m = lhsCubeType.getVectorLength();
    auto k = lhsCubeType.getHeight();
    auto n = rhsCubeType.getHeight();

    auto lhsLayerSliceType = SliceType::get(ctx, bitwidth, m);
    auto rhsLayerSliceType = SliceType::get(ctx, bitwidth, k);

    Value resBitwidthVal = rewriter.create<arith::ConstantIntOp>(loc, bitwidth * 2, 64);
    Value mVal = rewriter.create<arith::ConstantIntOp>(loc, m, 64);
    Value nVal = rewriter.create<arith::ConstantIntOp>(loc, n, 64);

    auto resCubeType = CubeType::get(ctx, bitwidth * 2, m, n);
    Value resCube = rewriter.create<CreateCubeOp>(loc, resCubeType, resBitwidthVal, mVal, nVal);
    auto resSliceType = SliceType::get(ctx, bitwidth * 2, m);

    Value c0Val = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value layerNumVal = rewriter.create<arith::ConstantIndexOp>(loc, n);
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kIndexVal = rewriter.create<arith::ConstantIndexOp>(loc, k);

    auto outerloop = rewriter.create<scf::ForOp>(loc, c0Val, layerNumVal, c1Val, ValueRange{resCube},
        [&](OpBuilder &outerBuilder, Location outerLoc, Value outerIV, ValueRange outerIterArgs){
          Value curCube = outerIterArgs[0];
          Value outerIVI64 = outerBuilder.create<arith::IndexCastOp>(
              outerLoc, outerBuilder.getI64Type(), outerIV);
          
          Value layerSlice = outerBuilder.create<ExtractSliceOp>(outerLoc, rhsLayerSliceType, rhsCube, outerIVI64);

          Value resSlice = rewriter.create<CreateSliceOp>(loc, resSliceType, resBitwidthVal, mVal);
          auto loop = rewriter.create<scf::ForOp>(outerLoc, c0Val, kIndexVal, c1Val,
              ValueRange{resSlice},
              [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs){
                Value curRes = iterArgs[0];
                Value ivI64 = builder.create<arith::IndexCastOp>(
                    bodyLoc, builder.getI64Type(), iv);

                Value mulLhs = builder.create<ExtractSliceOp>(bodyLoc, lhsLayerSliceType, lhsCube, ivI64);
                Value mulRhs = builder.create<ExtractAndReplicateColumnOp>(bodyLoc, lhsLayerSliceType, layerSlice, ivI64, mVal);
                Value product = builder.create<bits::MulIOp>(bodyLoc, resSliceType, mulLhs, mulRhs);

                Value accSlice = builder.create<bits::AddIOp>(bodyLoc, resSliceType, curRes, product);

                builder.create<scf::YieldOp>(bodyLoc, accSlice);
              });
          Value newCube = outerBuilder.create<InsertSliceOp>(outerLoc, resCubeType, curCube, loop.getResult(0), outerIVI64);
          outerBuilder.create<scf::YieldOp>(outerLoc, newCube);
        });

    rewriter.replaceOp(op, outerloop.getResults());

    return success();
  }
};

struct LowerMulIPattern :  public OpRewritePattern<MulIOp> {
  using OpRewritePattern<MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulIOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value lhsSlice = op.getLhs();
    Value rhsSlice = op.getRhs();

    auto lhsType = cast<SliceType>(lhsSlice.getType());
    auto rhsType = cast<SliceType>(rhsSlice.getType());
    if (!lhsType || !rhsType || lhsType.getBitWidth() != rhsType.getBitWidth())
      return failure();

    Value cstint1  = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    auto operandBitWidth = lhsType.getBitWidth();
    auto vecLen = lhsType.getVectorLength();
    Value bitwidthVal = rewriter.create<arith::ConstantIntOp>(loc, operandBitWidth, 64);
    Value productBitwidthVal = rewriter.create<arith::ConstantIntOp>(loc, operandBitWidth * 2, 64);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);

    auto rowSliceType = SliceType::get(ctx, 1, vecLen);
    Value all1Row = rewriter.create<CreateAllOneSliceOp>(loc, rowSliceType, cstint1, vecLenVal);

    Value c0Val = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value bitwidthIndexVal = rewriter.create<arith::ConstantIndexOp>(loc, operandBitWidth);

    Value all0Slice = rewriter.create<CreateSliceOp>(loc, lhsType, bitwidthVal, vecLenVal);
    auto resSliceType = SliceType::get(ctx, operandBitWidth * 2, vecLen);
    Value resSlice = rewriter.create<CreateSliceOp>(loc, resSliceType, productBitwidthVal, vecLenVal);

    auto loop = rewriter.create<scf::ForOp>(loc, c0Val, bitwidthIndexVal, c1Val, ValueRange{resSlice},
        [&](OpBuilder &builder, Location bodyLoc, Value iv, ValueRange iterArgs){
          Value curRes = iterArgs[0];
          Value ivI64 = builder.create<arith::IndexCastOp>(bodyLoc, builder.getI64Type(), iv);
          Value numToShift = builder.create<arith::AddIOp>(bodyLoc, ivI64, cstint1);

          Value curRow = builder.create<ExtractRowSliceOp>(bodyLoc, rowSliceType, rhsSlice, ivI64);
          Value maskRow = builder.create<AndOp>(bodyLoc, rowSliceType, curRow, all1Row);
          Value selected = builder.create<MuxOp>(bodyLoc, lhsType, lhsSlice, all0Slice, maskRow);
          Value extended = builder.create<ExtensionIOp>(bodyLoc, resSliceType, selected, bitwidthVal);
          Value shifted = builder.create<ShiftUpOp>(bodyLoc, resSliceType, extended, numToShift);

          Value partialRes = builder.create<AddIOp>(bodyLoc, resSliceType, curRes, shifted);
          builder.create<scf::YieldOp>(bodyLoc, partialRes);
        });

    rewriter.replaceOp(op, loop.getResults());

    return success();
  }
};

struct BitsCubeToSlicePass
    : public impl::BitsCubeToSlicePassBase<BitsCubeToSlicePass> {

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LowerMatvecIPattern, LowerMatmulIPattern, LowerMulIPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::bits
