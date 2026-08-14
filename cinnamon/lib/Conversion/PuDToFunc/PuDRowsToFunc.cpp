//===- PuDRowsToFunc.cpp - PuD row ops to guest runtime calls -------------===//
//
// Lowers the PuD row ops to calls into the guest runtime, leaving the loop
// structure alone so the rest of the pipeline can take it to LLVM IR normally.
//
// The ABI is four calls:
//
//   void *pud_row     (int64_t channel, int64_t rank, int64_t bank,
//                      int64_t subarray, int64_t row, int32_t group)
//   void *pud_ctrl_row(int64_t channel, int64_t rank, int64_t bank,
//                      int64_t subarray, int32_t ctrlRow)
//   void  rowop_ap    (void *dst)
//   void  rowop_aap   (void *dst, void *src)
//
// The two address calls are where a location becomes something a target can
// execute against: the dialect says which row of which bank, and the target
// runtime, which owns the memory, says where that is. Control rows go through
// their own call because they are named rather than numbered -- the substrate
// reserves them, so the runtime knows where they are without being told.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/PuDToFunc/PuDToFunc.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

using namespace mlir;
using namespace mlir::pud;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/PuDPasses.h.inc"

namespace {

/// Declares a callee once per module.
func::FuncOp getOrDeclare(ModuleOp module, StringRef name, FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

/// A data row: the full coordinate, plus which group of the row space it is in.
struct GetRowLowering : public OpConversionPattern<GetRowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto i64 = rewriter.getI64Type();
    auto i32 = rewriter.getI32Type();
    auto type =
        rewriter.getFunctionType({i64, i64, i64, i64, i64, i32}, {ptrType});
    func::FuncOp callee = getOrDeclare(module, "pud_row", type);

    auto rowType = cast<RowType>(op.getAddr().getType());
    Value group = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), rowType.getGroup(), 32);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee,
        ValueRange{adaptor.getChannelID(), adaptor.getRankID(),
                   adaptor.getBankID(), adaptor.getSubarrayID(),
                   adaptor.getRowID(), group});
    return success();
  }
};

/// A control row: named rather than numbered, so the name is what crosses.
struct GetCtrlRowLowering : public OpConversionPattern<GetCtrlRowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetCtrlRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto i64 = rewriter.getI64Type();
    auto i32 = rewriter.getI32Type();
    auto type = rewriter.getFunctionType({i64, i64, i64, i64, i32}, {ptrType});
    func::FuncOp callee = getOrDeclare(module, "pud_ctrl_row", type);

    Value which = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), static_cast<int64_t>(op.getCtrlRow()), 32);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee,
        ValueRange{adaptor.getChannelID(), adaptor.getRankID(),
                   adaptor.getBankID(), adaptor.getSubarrayID(), which});
    return success();
  }
};

struct AAPLowering : public OpConversionPattern<AAPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto type = rewriter.getFunctionType({ptrType, ptrType}, {});
    func::FuncOp callee = getOrDeclare(module, "rowop_aap", type);
    // the guest stub takes (dst, src); the op names them the other way round
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee, ValueRange{adaptor.getDstAddr(), adaptor.getSrcAddr()});
    return success();
  }
};

struct APLowering : public OpConversionPattern<APOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(APOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto type = rewriter.getFunctionType({ptrType}, {});
    func::FuncOp callee = getOrDeclare(module, "rowop_ap", type);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee,
                                              ValueRange{adaptor.getAddr()});
    return success();
  }
};

struct ConvertPuDRowsToFunc
    : public ConvertPuDRowsToFuncBase<ConvertPuDRowsToFunc> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    // a row address is just a pointer once the runtime has resolved it
    converter.addConversion([ctx](RowType) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });

    ConversionTarget target(*ctx);
    target.addIllegalDialect<PuDDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect, LLVM::LLVMDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<GetRowLowering, GetCtrlRowLowering, AAPLowering, APLowering>(
        converter, ctx);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pud::createConvertPuDRowsToFuncPass() {
  return std::make_unique<ConvertPuDRowsToFunc>();
}
