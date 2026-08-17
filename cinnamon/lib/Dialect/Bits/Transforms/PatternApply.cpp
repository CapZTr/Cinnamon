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

/// Where a bits.row_add keeps its carry. An adder whose carry stays in the
/// dual-contact cell it is computed in costs one command less per bit than one
/// that has to read it from a row and write it back, so which of the two an
/// expansion builds is a property of the expansion, decided here and recorded
/// rather than guessed at further down.
constexpr llvm::StringLiteral kCarryAttrName = "bits.carry";
constexpr llvm::StringLiteral kCarryResident = "resident";
constexpr llvm::StringLiteral kCarryLoaded = "loaded";

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
          // The carry is this loop's only loop-carried row and starts at zero,
          // so it never has to reach a data row: it stays in the dual-contact
          // cell the adder computes it in, seeded once ahead of the loop. That
          // is the whole difference between 8 commands a bit and 9.
          rowAdd->setAttr(kCarryAttrName, builder.getStringAttr(kCarryResident));

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

/// Subtraction: the same shape as AddIPattern, starting its chain at one.
///
/// `lhs - rhs = lhs + ~rhs + 1`, and on this substrate that +1 is the whole
/// difference in the loop's setup: the borrow row is seeded from the row held at
/// one instead of the row held at zero. Complementing the subtrahend is not
/// spelled here at all, because it is not a step -- `bits.row_sub` reads the
/// other contact of the cell it parks the subtrahend in. 9n + 1, as
/// getRecipeForSub prices it.
struct SubIPattern : public OpRewritePattern<SubIOp> {
  using OpRewritePattern<SubIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubIOp op,
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

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value bitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, bitwidth);

    Value bitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, bitwidth, 64);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);

    auto rowType = BitRowType::get(ctx, vecLen);
    // The +1 of ~rhs + 1.
    Value borrowRow =
        rewriter.create<CreateAllOneRowOp>(loc, rowType, vecLenVal);
    Value resSlice =
        rewriter.create<CreateSliceOp>(loc, sliceType, bitwidthVal, vecLenVal);

    auto loop = rewriter.create<scf::ForOp>(
        loc, c0Index, bitwidthIndex, c1Index, ValueRange{resSlice, borrowRow},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value curSlice = iterArgs[0];
          Value binRow = iterArgs[1];
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);

          Value lhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, lhsSlice, ivI64);
          Value rhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, rhsSlice, ivI64);
          auto rowSub = builder.create<RowSubOp>(bodyLoc, rowType, rowType,
                                                 lhsRow, rhsRow, binRow);
          // Same reasoning as the adder: the borrow is the loop's only carried
          // row and starts from a constant, so it never leaves the dual-contact
          // cell it is computed in.
          rowSub->setAttr(kCarryAttrName, builder.getStringAttr(kCarryResident));

          Value updatedSlice = builder.create<InsertRowOp>(
              bodyLoc, sliceType, curSlice, rowSub.getDiff(), ivI64);
          builder.create<scf::YieldOp>(
              bodyLoc, ValueRange{updatedSlice, rowSub.getBout()});
        });

    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

/// Widening add: the same adder, keeping the carry-out instead of dropping it.
///
/// A truncating add throws away the carry that falls off the top. This one is
/// what a reduction tree wants, so it keeps it as one more result row -- and
/// that store is a command, because the carry lives in a dual-contact cell that
/// the next addition's carry-init would overwrite. 8n + 2, as getRecipeForAdd
/// prices a widening add.
struct AddIFullPattern : public OpRewritePattern<AddIFullOp> {
  using OpRewritePattern<AddIFullOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddIFullOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value lhsSlice = op.getLhs();
    Value rhsSlice = op.getRhs();

    auto operandType = dyn_cast<SliceType>(lhsSlice.getType());
    auto resType = dyn_cast<SliceType>(op.getResult().getType());
    if (!operandType || !resType)
      return failure();

    auto bitwidth = operandType.getBitWidth();
    auto vecLen = operandType.getVectorLength();

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value bitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, bitwidth);
    Value carryRowIdx =
        rewriter.create<arith::ConstantIntOp>(loc, bitwidth, 64);

    Value resBitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, resType.getBitWidth(), 64);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);

    auto rowType = BitRowType::get(ctx, vecLen);
    Value carryRow =
        rewriter.create<CreateAllZeroRowOp>(loc, rowType, vecLenVal);
    Value resSlice =
        rewriter.create<CreateSliceOp>(loc, resType, resBitwidthVal, vecLenVal);

    auto loop = rewriter.create<scf::ForOp>(
        loc, c0Index, bitwidthIndex, c1Index, ValueRange{resSlice, carryRow},
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
          rowAdd->setAttr(kCarryAttrName, builder.getStringAttr(kCarryResident));

          Value updatedSlice = builder.create<InsertRowOp>(
              bodyLoc, resType, curSlice, rowAdd.getSum(), ivI64);
          builder.create<scf::YieldOp>(
              bodyLoc, ValueRange{updatedSlice, rowAdd.getCout()});
        });

    // The carry that fell off the top is the result's top row.
    Value withCarry = rewriter.create<InsertRowOp>(
        loc, resType, loop.getResult(0), loop.getResult(1), carryRowIdx);

    rewriter.replaceOp(op, withCarry);
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
          addLSB->setAttr(kCarryAttrName, builder.getStringAttr(kCarryLoaded));
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
                innerAdd->setAttr(kCarryAttrName,
                                  innerBuilder.getStringAttr(kCarryLoaded));

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
          outerAdd->setAttr(kCarryAttrName, builder.getStringAttr(kCarryLoaded));
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
    rowAdd->setAttr(kCarryAttrName, rewriter.getStringAttr(kCarryLoaded));
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
          finalLoopAdd->setAttr(kCarryAttrName,
                                builder.getStringAttr(kCarryLoaded));
        //   Value resRowIndex = builder.create<arith::AddIOp>(
        //       bodyLoc, partialRowIndex, partialAddBitwidthVal);
          Value resRowIndex = builder.create<arith::AddIOp>(
              bodyLoc, ivI64, mrBitwidthIndexV);
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
    addAfterFinalLoop->setAttr(kCarryAttrName,
                               rewriter.getStringAttr(kCarryLoaded));
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

/// Min and max: compare the whole operand, then select from it.
///
/// Two rolled loops rather than one, because a comparison has to finish before a
/// select can start -- the verdict is a property of the whole operand, and the
/// most significant bit is what settles it. 13n + 2, which is 3n for the
/// comparison, 10n for the select, one command to seed the verdict and one to
/// move it into the cell the select reads it and its complement from.
///
/// This deliberately does not use the MinMax row-op template, which emits a min
/// and a max together from a verdict handed to it: `bits.min` is one output, so
/// the fused form would produce a result nothing asked for and, more to the
/// point, would still need the comparison it does not contain.
template <typename Op, bool IsMax>
struct MinMaxPattern : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();
    Value lhsSlice = op.getLhs();
    Value rhsSlice = op.getRhs();

    auto sliceType = dyn_cast<SliceType>(lhsSlice.getType());
    auto resType = dyn_cast<SliceType>(op.getResult().getType());
    if (!sliceType || !resType)
      return failure();

    int64_t bitwidth = sliceType.getBitWidth();
    int64_t vecLen = sliceType.getVectorLength();

    Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value bitwidthIndex =
        rewriter.create<arith::ConstantIndexOp>(loc, bitwidth);
    Value bitwidthVal =
        rewriter.create<arith::ConstantIntOp>(loc, bitwidth, 64);
    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);

    auto rowType = BitRowType::get(ctx, vecLen);
    // Seeded false, so operands that compare equal all the way up select the
    // right-hand one -- which is what makes min and max agree on ties.
    Value verdict =
        rewriter.create<CreateAllZeroRowOp>(loc, rowType, vecLenVal);

    auto compare = rewriter.create<scf::ForOp>(
        loc, c0Index, bitwidthIndex, c1Index, ValueRange{verdict},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);
          Value lhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, lhsSlice, ivI64);
          Value rhsRow =
              builder.create<ExtractRowOp>(bodyLoc, rowType, rhsSlice, ivI64);
          auto step = builder.create<RowGreaterStepOp>(bodyLoc, rowType, lhsRow,
                                                       rhsRow, iterArgs[0]);
          step->setAttr(kCarryAttrName, builder.getStringAttr(kCarryResident));
          builder.create<scf::YieldOp>(bodyLoc, step.getGOut());
        });

    Value resSlice =
        rewriter.create<CreateSliceOp>(loc, resType, bitwidthVal, vecLenVal);
    // max picks the left operand when it is the greater one, min the right.
    Value takeIfGreater = IsMax ? lhsSlice : rhsSlice;
    Value takeOtherwise = IsMax ? rhsSlice : lhsSlice;

    auto select = rewriter.create<scf::ForOp>(
        loc, c0Index, bitwidthIndex, c1Index, ValueRange{resSlice},
        [&](OpBuilder &builder, Location bodyLoc, Value iv,
            ValueRange iterArgs) {
          Value ivI64 = builder.create<arith::IndexCastOp>(
              bodyLoc, builder.getI64Type(), iv);
          Value xRow = builder.create<ExtractRowOp>(bodyLoc, rowType,
                                                    takeIfGreater, ivI64);
          Value yRow = builder.create<ExtractRowOp>(bodyLoc, rowType,
                                                    takeOtherwise, ivI64);
          Value picked = builder.create<RowSelectOp>(
              bodyLoc, rowType, xRow, yRow, compare.getResult(0));
          Value updated = builder.create<InsertRowOp>(
              bodyLoc, resType, iterArgs[0], picked, ivI64);
          builder.create<scf::YieldOp>(bodyLoc, updated);
        });

    rewriter.replaceOp(op, select.getResult(0));
    return success();
  }
};

/// Range scan: one step per bit plane, unrolled, then one fold.
///
/// Unrolled rather than looped because the bound bits pick which of the four
/// substrate sequences a bit position runs, and consecutive positions rarely
/// agree. The bit widths this op is written for are 8 to 32, so the unrolled
/// body is a few hundred commands -- and the loop it replaces would have had to
/// carry the bounds in rows to stay rolled, which is exactly the cost the
/// four-variant design exists to avoid.
///
/// Both chains are seeded to one: the scan is inclusive at both ends, which is
/// what the two chains are defined as (">= lower" and "<= upper"). The seeds do
/// not reach a data row; the lowering turns them into the two copies that put
/// them in their cells.
struct RangeScanPattern : public OpRewritePattern<RangeScanOp> {
  using OpRewritePattern<RangeScanOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RangeScanOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto ctx = rewriter.getContext();

    auto inputType = dyn_cast<SliceType>(op.getInput().getType());
    auto resultType = dyn_cast<SliceType>(op.getResult().getType());
    if (!inputType || !resultType)
      return failure();

    int64_t bitwidth = inputType.getBitWidth();
    int64_t vecLen = inputType.getVectorLength();
    if (bitwidth <= 0 || bitwidth > 64)
      return failure();

    uint64_t lower = static_cast<uint64_t>(op.getLower());
    uint64_t upper = static_cast<uint64_t>(op.getUpper());

    Value vecLenVal = rewriter.create<arith::ConstantIntOp>(loc, vecLen, 64);
    Value resWidthVal = rewriter.create<arith::ConstantIntOp>(
        loc, resultType.getBitWidth(), 64);
    auto rowType = BitRowType::get(ctx, vecLen);

    // The mask's slice is created before the fold that writes into it: row
    // placement reads a result's home off the consumer that names it, and a
    // consumer whose slice is not bound yet cannot name one, which would send
    // the mask through a scratch row and cost a copy.
    Value maskSlice =
        rewriter.create<CreateSliceOp>(loc, resultType, resWidthVal, vecLenVal);

    Value ge = rewriter.create<CreateAllOneRowOp>(loc, rowType, vecLenVal);
    Value le = rewriter.create<CreateAllOneRowOp>(loc, rowType, vecLenVal);

    for (int64_t j = 0; j < bitwidth; j++) {
      Value idx = rewriter.create<arith::ConstantIntOp>(loc, j, 64);
      Value valueRow =
          rewriter.create<ExtractRowOp>(loc, rowType, op.getInput(), idx);
      auto step = rewriter.create<RowScanStepOp>(
          loc, rowType, rowType, valueRow, ge, le,
          rewriter.getBoolAttr((lower >> j) & 1),
          rewriter.getBoolAttr((upper >> j) & 1));
      ge = step.getGeOut();
      le = step.getLeOut();
    }

    Value mask = rewriter.create<RowScanFoldOp>(loc, rowType, ge, le);
    Value zeroIdx = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value result = rewriter.create<InsertRowOp>(loc, resultType, maskSlice,
                                                mask, zeroIdx);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BitsPatternApplyPass
    : public impl::BitsPatternApplyPassBase<BitsPatternApplyPass> {
  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    constexpr bool useReadableMulIPattern = false;
    if (useReadableMulIPattern) {
      patterns
          .add<AddIPattern, AddIFullPattern, SubIPattern, RangeScanPattern,
               MinMaxPattern<MaxOp, true>, MinMaxPattern<MinOp, false>,
               MulIReadablePattern,
               RowWiseLogicPattern<AndOp, RowAndOp>,
               RowWiseLogicPattern<OrOp, RowOrOp>,
               RowWiseLogicPattern<XOrOp, RowXOrOp>,
               RowWiseLogicPattern<XNOrOp, RowXNorOp>>(ctx);
    } else {
      patterns.add<AddIPattern, AddIFullPattern, SubIPattern, RangeScanPattern,
                   MinMaxPattern<MaxOp, true>, MinMaxPattern<MinOp, false>,
                   MulIPattern,
                   RowWiseLogicPattern<AndOp, RowAndOp>,
                   RowWiseLogicPattern<OrOp, RowOrOp>,
                   RowWiseLogicPattern<XOrOp, RowXOrOp>,
               RowWiseLogicPattern<XNOrOp, RowXNorOp>>(ctx);
    }
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::bits
