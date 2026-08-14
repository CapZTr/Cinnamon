//===- PuDRowsToFunc.cpp - PuD row ops to guest runtime calls -------------===//
//
// Lowers the PuD row ops to calls into the guest runtime, leaving the loop
// structure alone so the rest of the pipeline can take it to LLVM IR normally.
//
// The three callees are the whole ABI:
//
//   void *pud_row_addr(int64_t slot, int64_t bank)
//   void  rowop_ap (void *dst)
//   void  rowop_aap(void *dst, void *src)
//
// `rowop_ap` and `rowop_aap` are the existing guest stubs -- two-byte opcodes
// the CPU decodes into row-op microops, with their pointer arguments arriving in
// the usual argument registers. `pud_row_addr` is the runtime's own: row slots
// are allocated as row-aligned blocks, so only the program that allocated them
// can turn a (slot, bank) pair into an address.
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

struct RowAddrLowering : public OpConversionPattern<RowAddrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RowAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto type = rewriter.getFunctionType({rewriter.getI64Type(),
                                          rewriter.getI64Type()},
                                         {ptrType});
    func::FuncOp callee = getOrDeclare(module, "pud_row_addr", type);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee, ValueRange{adaptor.getSlot(), adaptor.getBank()});
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
    patterns.add<RowAddrLowering, AAPLowering, APLowering>(converter, ctx);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pud::createConvertPuDRowsToFuncPass() {
  return std::make_unique<ConvertPuDRowsToFunc>();
}
