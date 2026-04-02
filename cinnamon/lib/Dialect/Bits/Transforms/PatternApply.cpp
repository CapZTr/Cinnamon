#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace mlir::bits {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSPATTERNAPPLYPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

template <typename SliceOp, typename RowOp>
struct RowWiseLogicPattern : public OpRewritePattern<SliceOp> {
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value lhsSlice = op.getLhs();
    Value rhsSlice = op.getRhs();

    auto lhsSliceType = dyn_cast<SliceType>(lhsSlice.getType());
    auto rhsSliceType = dyn_cast<SliceType>(rhsSlice.getType());
    auto resSliceType = dyn_cast<SliceType>(op.getResult().getType());
    if (!lhsSliceType || !rhsSliceType || !resSliceType)
      return failure();

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value bitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, resSliceType.getBitWidth());

    Value bitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, resSliceType.getBitWidth(), 64);
    Value vecLenVal =
        rewriter.create<arith::ConstantIntOp>(loc, resSliceType.getVectorLength(), 64);

    auto rowType = BitRowType::get(ctx, resSliceType.getVectorLength());
    Value initResSlice = rewriter.create<CreateSliceOp>(
        loc, resSliceType, bitwidthVal, vecLenVal);

    auto loop = rewriter.create<scf::ForOp>(
        loc, c0Index, bitwidthIndex, c1Index, ValueRange{initResSlice},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curResSlice = iterArgs[0];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value lhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, lhsSlice, ivI64);
          Value rhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, rhsSlice, ivI64);
          Value logicRow =
              builder.create<RowOp>(bodyLoc, rowType, lhsRow, rhsRow);

          Value updatedResSlice = builder.create<InsertRowOp>(
              bodyLoc, resSliceType, curResSlice, logicRow, ivI64);

          builder.create<scf::YieldOp>(bodyLoc, updatedResSlice);
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

struct AddIPattern : public OpRewritePattern<AddIOp> {
  using OpRewritePattern<AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value lhsSlice = op.getLhs();
    Value rhsSlice = op.getRhs();

    auto sliceType = dyn_cast<SliceType>(lhsSlice.getType());
    if (!sliceType)
      return failure();

    auto bitwidth = sliceType.getBitWidth();
    auto vecLen = sliceType.getVectorLength();

    Value c0Val = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value bitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, bitwidth);
    Value c1Val = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    Value bitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, bitwidth, 64);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);
    auto rowType = BitRowType::get(ctx, vecLen);
    Value carryRow =
        rewriter.create<CreateAllZeroRowOp>(loc, rowType, vecLenVal);
    Value resSlice =
        rewriter.create<CreateSliceOp>(loc, sliceType, bitwidthVal, vecLenVal);

    auto loop = rewriter.create<scf::ForOp>(
        loc, c0Val, bitwidthIndex, c1Val, ValueRange{resSlice, carryRow},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curSlice = iterArgs[0];
          Value cinRow = iterArgs[1];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value lhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, lhsSlice, ivI64);
          Value rhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, rhsSlice, ivI64);
          auto rowAdd = builder.create<RowAddOp>(bodyLoc, rowType, rowType,
                                                 lhsRow, rhsRow, cinRow);

          Value sumRow = rowAdd.getSum();
          Value coutRow = rowAdd.getCout();

          Value updatedSlice = builder.create<InsertRowOp>(
              bodyLoc, sliceType, curSlice, sumRow, ivI64);

          builder.create<scf::YieldOp>(bodyLoc,
                                       ValueRange{updatedSlice, coutRow});
        });

    rewriter.replaceOp(op, loop.getResult(0));

    return success();
  }
};

struct MulIPattern : public OpRewritePattern<MulIOp> {
  using OpRewritePattern<MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value mdSlice = op.getLhs();
    Value mrSlice = op.getRhs();

    auto mdSliceType = dyn_cast<SliceType>(mdSlice.getType());
    auto mrSliceType = dyn_cast<SliceType>(mrSlice.getType());
    auto resSliceType = dyn_cast<SliceType>(op.getResult().getType());
    if (!mdSliceType || !mrSliceType || !resSliceType)
      return failure();

    auto vecLen = mdSliceType.getVectorLength();
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);

    auto rowType = BitRowType::get(ctx, vecLen);
    Value c0Row = rewriter.create<CreateAllZeroRowOp>(loc, rowType, vecLenVal);

    auto mdBitwidth = mdSliceType.getBitWidth();
    auto partialAddBitwidth = mdBitwidth - 1;
    auto mrBitwidth = mrSliceType.getBitWidth();
    auto resBitwidth = resSliceType.getBitWidth();

    Value resBitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, resBitwidth, 64);

    Value resSlice = rewriter.create<CreateSliceOp>(loc, resSliceType,
                                                    resBitwidthVal, vecLenVal);

    Value partialAddBitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, partialAddBitwidth, 64);
    auto partialAddSliceType = SliceType::get(ctx, partialAddBitwidth, vecLen);

    Value c0Val = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value c1Val = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    Value mdBitwidthIndexV =
        rewriter.create<arith::ConstantIntOp>(loc, mdBitwidth - 1, 64);
    Value mdMSBRow =
        rewriter.create<ExtractRowOp>(loc, rowType, mdSlice, mdBitwidthIndexV);
    Value mdLSBRow =
        rewriter.create<ExtractRowOp>(loc, rowType, mdSlice, c0Val);
    Value mrLSBRow =
        rewriter.create<ExtractRowOp>(loc, rowType, mrSlice, c0Val);
    Value resLSBRow =
        rewriter.create<RowAndOp>(loc, rowType, mdLSBRow, mrLSBRow);
    Value mrBitwidthIndexV =
        rewriter.create<arith::ConstantIntOp>(loc, mrBitwidth - 1, 64);
    Value mrMSBRow =
        rewriter.create<ExtractRowOp>(loc, rowType, mrSlice, mrBitwidthIndexV);
    Value updatedResSlice1 = rewriter.create<InsertRowOp>(
        loc, resSliceType, resSlice, resLSBRow, c0Val);

    Value partialAddSlice = rewriter.create<CreateSliceOp>(
        loc, partialAddSliceType, partialAddBitwidthVal, vecLenVal);

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value partialAddBitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, partialAddBitwidth);

    auto loop1 = rewriter.create<scf::ForOp>(
        loc, c0Index, partialAddBitwidthIndex, c1Index,
        ValueRange{partialAddSlice},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curSlice = iterArgs[0];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);
          Value mdIndexVal =
              builder.create<arith::AddIOp>(bodyLoc, ivI64, c1Val);

          Value mdRow = builder.create<ExtractRowOp>(bodyLoc, rowType, mdSlice,
                                                     mdIndexVal);
          Value andResRow =
              builder.create<RowAndOp>(bodyLoc, rowType, mdRow, mrLSBRow);
          Value updatedPartialSlice = builder.create<InsertRowOp>(
              bodyLoc, partialAddSliceType, curSlice, andResRow, ivI64);
          builder.create<scf::YieldOp>(bodyLoc, updatedPartialSlice);
        });

    Value partialAddSlice1 = loop1.getResult(0);
    Value mainLoopIterNum =
        rewriter.create<arith::ConstantIndexOp>(loc, mrBitwidth - 1);
    Value mainLoopCarry = c0Row;
    auto loop2 = rewriter.create<scf::ForOp>(
        loc, c1Index, mainLoopIterNum, c1Index,
        ValueRange{updatedResSlice1, partialAddSlice1, mainLoopCarry},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curResSlice = iterArgs[0];
          Value curPartialSlice = iterArgs[1];
          Value finalCin = iterArgs[2];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value mrRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, mrSlice, ivI64);
          Value andFirstRow =
              builder.create<RowAndOp>(bodyLoc, rowType, mdLSBRow, mrRow);
          Value curPartialLSB = builder.create<ExtractRowOp>(
              bodyLoc, rowType, curPartialSlice, c0Val);
          auto addLSB = builder.create<RowAddOp>(
              bodyLoc, rowType, rowType, andFirstRow, curPartialLSB, c0Row);
          Value lsbCout = addLSB.getCout();
          Value lsbSum = addLSB.getSum();
          Value updatedResSlice2 = builder.create<InsertRowOp>(
              bodyLoc, resSliceType, curResSlice, lsbSum, ivI64);
          auto loop3 = rewriter.create<scf::ForOp>(
              bodyLoc, c1Index, partialAddBitwidthIndex, c1Index,
              ValueRange{curPartialSlice, lsbCout},
              [&](OpBuilder &innerBuilder, Location innerBodyLoc, Value innerIv,
                  ValueRange innerIterArgs) {
                Value curInnerPartialSlice = innerIterArgs[0];
                Value curCarry = innerIterArgs[1];
                Value innerIvI64 = innerBuilder.create<arith::IndexCastOp>(
                    innerBodyLoc, innerBuilder.getI64Type(), innerIv);
                Value partialRowIndex = innerBuilder.create<arith::SubIOp>(
                    innerBodyLoc, innerIvI64, c1Val);

                Value mdRow = innerBuilder.create<ExtractRowOp>(
                    innerBodyLoc, rowType, mdSlice, innerIvI64);
                Value andRow = innerBuilder.create<RowAndOp>(
                    innerBodyLoc, rowType, mdRow, mrRow);
                Value curPartialRow = innerBuilder.create<ExtractRowOp>(
                    innerBodyLoc, rowType, curInnerPartialSlice, innerIvI64);
                auto innerAdd = innerBuilder.create<RowAddOp>(
                    innerBodyLoc, rowType, rowType, andRow, curPartialRow,
                    curCarry);

                Value innerAddSum = innerAdd.getSum();
                Value innerCout = innerAdd.getCout();

                Value updatedPartialSlice = innerBuilder.create<InsertRowOp>(
                    innerBodyLoc, partialAddSliceType, curInnerPartialSlice,
                    innerAddSum, partialRowIndex);
                innerBuilder.create<scf::YieldOp>(
                    innerBodyLoc, ValueRange{updatedPartialSlice, innerCout});
              });
          Value partialSlice = loop3.getResult(0);
          Value carry = loop3.getResult(1);
          Value outerAnd =
              builder.create<RowAndOp>(bodyLoc, rowType, mdMSBRow, mrRow);
          auto outerAdd = builder.create<RowAddOp>(bodyLoc, rowType, rowType,
                                                   outerAnd, carry, finalCin);
          Value updatedPartialSlice = builder.create<InsertRowOp>(
              bodyLoc, partialAddSliceType, partialSlice, outerAdd.getSum(),
              partialAddBitwidthVal);
          builder.create<scf::YieldOp>(bodyLoc, ValueRange{updatedResSlice2,
                                                           updatedPartialSlice,
                                                           outerAdd.getCout()});
        });

    Value curResSlice = loop2.getResult(0);
    Value curPartialSlice = loop2.getResult(1);
    Value mainLoopCout = loop2.getResult(2);

    Value afterMainLoopAnd =
        rewriter.create<RowAndOp>(loc, rowType, mdLSBRow, mrMSBRow);
    Value curPartialLSB =
        rewriter.create<ExtractRowOp>(loc, rowType, curPartialSlice, c0Val);
    auto rowAdd = rewriter.create<RowAddOp>(
        loc, rowType, rowType, afterMainLoopAnd, curPartialLSB, c0Row);
    Value updatedResSlice3 = rewriter.create<InsertRowOp>(
        loc, resSliceType, curResSlice, rowAdd.getSum(), mrBitwidthIndexV);
    Value finalCarry = rowAdd.getCout();

    auto finalLoop = rewriter.create<scf::ForOp>(
        loc, c1Index, partialAddBitwidthIndex, c1Index,
        ValueRange{updatedResSlice3, finalCarry},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curResSlice = iterArgs[0];
          Value currCin = iterArgs[1];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value mdRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, mdSlice, ivI64);
          Value finalLoopAnd =
              builder.create<RowAndOp>(bodyLoc, rowType, mdRow, mrMSBRow);
          Value partialRowIndex =
              builder.create<arith::SubIOp>(bodyLoc, ivI64, c1Val);
          Value partialRow = builder.create<ExtractRowOp>(
              bodyLoc, rowType, curPartialSlice, partialRowIndex);
          auto finalLoopAdd = builder.create<RowAddOp>(
              bodyLoc, rowType, rowType, finalLoopAnd, partialRow, currCin);
          Value resRowIndex = builder.create<arith::AddIOp>(
              bodyLoc, partialRowIndex, partialAddBitwidthVal);
          Value updatedResSlice4 =
              builder.create<InsertRowOp>(bodyLoc, resSliceType, curResSlice,
                                          finalLoopAdd.getSum(), resRowIndex);
          builder.create<scf::YieldOp>(
              bodyLoc, ValueRange{updatedResSlice4, finalLoopAdd.getCout()});
        });

    Value resSliceAfterFianlLoop = finalLoop.getResult(0);
    Value coutFinalLoop = finalLoop.getResult(1);
    Value finalAnd =
        rewriter.create<RowAndOp>(loc, rowType, mdMSBRow, mrMSBRow);
    auto addAfterFinalLoop = rewriter.create<RowAddOp>(
        loc, rowType, rowType, finalAnd, mainLoopCout, coutFinalLoop);
    Value rowIdx =
        rewriter.create<arith::ConstantIntOp>(loc, resBitwidth - 2, 64);
    Value updatedResSlice5 =
        rewriter.create<InsertRowOp>(loc, resSliceType, resSliceAfterFianlLoop,
                                     addAfterFinalLoop.getSum(), rowIdx);
    Value resMSBIdx =
        rewriter.create<arith::ConstantIntOp>(loc, resBitwidth - 1, 64);
    Value finalProduct =
        rewriter.create<InsertRowOp>(loc, resSliceType, updatedResSlice5,
                                     addAfterFinalLoop.getCout(), resMSBIdx);

    rewriter.replaceOp(op, finalProduct);

    return success();
  }
};

struct MulIReadablePattern : public OpRewritePattern<MulIOp> {
  using OpRewritePattern<MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    Value mdSlice = op.getLhs();
    Value mrSlice = op.getRhs();

    auto mdSliceType = dyn_cast<SliceType>(mdSlice.getType());
    auto mrSliceType = dyn_cast<SliceType>(mrSlice.getType());
    auto resSliceType = dyn_cast<SliceType>(op.getResult().getType());
    if (!mdSliceType || !mrSliceType || !resSliceType)
      return failure();

    auto vecLen = mdSliceType.getVectorLength();
    auto mdBitwidth = mdSliceType.getBitWidth();
    auto mrBitwidth = mrSliceType.getBitWidth();
    auto resBitwidth = resSliceType.getBitWidth();

    auto rowType = BitRowType::get(ctx, vecLen);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);
    Value resBitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, resBitwidth, 64);
    Value zeroRow =
        rewriter.create<CreateAllZeroRowOp>(loc, rowType, vecLenVal);
    Value initRes = rewriter.create<CreateSliceOp>(loc, resSliceType,
                                                   resBitwidthVal, vecLenVal);

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value mdBitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, mdBitwidth);
    Value mrBitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, mrBitwidth);
    Value resBitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, resBitwidth);

    auto outerLoop = rewriter.create<scf::ForOp>(
        loc, c0Index, mrBitwidthIndex, c1Index, ValueRange{initRes},
        [&](OpBuilder &builder, Location bodyLoc, Value mrIdx,
            ValueRange outerIterArgs) {
          Value curRes = outerIterArgs[0];
          Value mrIdxI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), mrIdx);
          Value mrRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, mrSlice, mrIdxI64);

          auto mulAccLoop = builder.create<scf::ForOp>(
              bodyLoc, c0Index, mdBitwidthIndex, c1Index,
              ValueRange{curRes, zeroRow},
              [&](OpBuilder &innerBuilder, Location innerBodyLoc, Value mdIdx,
                  ValueRange mulAccIterArgs) {
                Value mulAccRes = mulAccIterArgs[0];
                Value carry = mulAccIterArgs[1];
                Value mdIdxI64 = innerBuilder.create<arith::IndexCastOp>(
                    innerBodyLoc, innerBuilder.getI64Type(), mdIdx);
                Value resIdxI64 = innerBuilder.create<arith::AddIOp>(
                    innerBodyLoc, mdIdxI64, mrIdxI64);

                Value mdRow = innerBuilder.create<ExtractRowOp>(
                    innerBodyLoc, rowType, mdSlice, mdIdxI64);
                Value andRow = innerBuilder.create<RowAndOp>(
                    innerBodyLoc, rowType, mdRow, mrRow);
                Value resRow = innerBuilder.create<ExtractRowOp>(
                    innerBodyLoc, rowType, mulAccRes, resIdxI64);
                auto rowAdd = innerBuilder.create<RowAddOp>(
                    innerBodyLoc, rowType, rowType, andRow, resRow, carry);
                Value updatedRes = innerBuilder.create<InsertRowOp>(
                    innerBodyLoc, resSliceType, mulAccRes, rowAdd.getSum(),
                    resIdxI64);
                innerBuilder.create<scf::YieldOp>(
                    innerBodyLoc, ValueRange{updatedRes, rowAdd.getCout()});
              });

          Value carryStartIndex =
              builder.create<arith::AddIOp>(bodyLoc, mrIdx, mdBitwidthIndex);
          auto carryPropLoop = builder.create<scf::ForOp>(
              bodyLoc, carryStartIndex, resBitwidthIndex, c1Index,
              ValueRange{mulAccLoop.getResult(0), mulAccLoop.getResult(1)},
              [&](OpBuilder &carryBuilder, Location carryBodyLoc, Value resIdx,
                  ValueRange carryIterArgs) {
                Value carryRes = carryIterArgs[0];
                Value carry = carryIterArgs[1];
                Value resIdxI64 = carryBuilder.create<arith::IndexCastOp>(
                    carryBodyLoc, carryBuilder.getI64Type(), resIdx);
                Value resRow = carryBuilder.create<ExtractRowOp>(
                    carryBodyLoc, rowType, carryRes, resIdxI64);
                auto rowAdd = carryBuilder.create<RowAddOp>(
                    carryBodyLoc, rowType, rowType, zeroRow, resRow, carry);
                Value updatedRes = carryBuilder.create<InsertRowOp>(
                    carryBodyLoc, resSliceType, carryRes, rowAdd.getSum(),
                    resIdxI64);
                carryBuilder.create<scf::YieldOp>(
                    carryBodyLoc, ValueRange{updatedRes, rowAdd.getCout()});
              });

          builder.create<scf::YieldOp>(bodyLoc, carryPropLoop.getResult(0));
        });

    rewriter.replaceOp(op, outerLoop.getResult(0));
    return success();
  }
};

struct BitsPatternApplyPass
    : public impl::BitsPatternApplyPassBase<BitsPatternApplyPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    constexpr bool useReadableMulIPattern = true;
    if (useReadableMulIPattern) {
      patterns
          .add<AddIPattern, MulIReadablePattern, RowWiseLogicPattern<AndOp, RowAndOp>,
               RowWiseLogicPattern<OrOp, RowOrOp>,
               RowWiseLogicPattern<XOrOp, RowXOrOp>>(ctx);
    } else {
      patterns.add<AddIPattern, MulIPattern,
                   RowWiseLogicPattern<AndOp, RowAndOp>,
                   RowWiseLogicPattern<OrOp, RowOrOp>,
                   RowWiseLogicPattern<XOrOp, RowXOrOp>>(ctx);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::bits
