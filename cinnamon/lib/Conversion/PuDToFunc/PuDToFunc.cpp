#include "cinm-mlir/Conversion/PuDToFunc/PuDToFunc.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>

#include <memory>

using namespace mlir;
using namespace mlir::pud;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/PuDPasses.h.inc>

namespace {

struct ConvertPuDToFunc
    : public ConvertPuDToFuncBase<ConvertPuDToFunc> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    // ctx->loadDialect<LLVM::LLVMDialect>();
    OpBuilder builder(ctx);

    const auto ptrType = LLVM::LLVMPointerType::get(ctx);

    ModuleOp module = func->getParentOfType<ModuleOp>();
    builder.setInsertionPointToStart(module.getBody());

    auto getOrCreateCallee = [&](StringRef name, FunctionType type)
        -> func::FuncOp {
      if (auto existing = module.lookupSymbol<func::FuncOp>(name))
        return existing;
      auto loc = module.getLoc();
      auto fn  = builder.create<func::FuncOp>(loc, name, type);
      fn.setPrivate();
      return fn;
    };

    auto pudGetRowType = builder.getFunctionType(TypeRange{}, TypeRange{ptrType});
    func::FuncOp pudGetRowFunc = getOrCreateCallee("pud_get_row", pudGetRowType);
    auto pudGetRowSym = SymbolRefAttr::get(pudGetRowFunc);

    auto pudApType = builder.getFunctionType(TypeRange{ptrType}, TypeRange{});
    func::FuncOp pudApFunc = getOrCreateCallee("pud_ap", pudApType);
    auto pudApSym = SymbolRefAttr::get(pudApFunc);

    auto pudAapType = builder.getFunctionType(TypeRange{ptrType, ptrType}, TypeRange{});
    func::FuncOp pudAapFunc = getOrCreateCallee("pud_aap", pudAapType);
    auto pudAapSym = SymbolRefAttr::get(pudAapFunc);

    auto newFuncType = builder.getFunctionType(TypeRange{}, TypeRange{});
    func.setType(newFuncType);

    Block &oldEntry = func.front();
    auto *newEntry  = new Block();
    func.getBody().push_back(newEntry);

    builder.setInsertionPointToStart(newEntry);
    IRMapping mapping;

    for (Operation &op : llvm::make_early_inc_range(oldEntry)) {
      if (isa<mlir::bits::TransposeOp>(&op) ||
          isa<mlir::bits::AssembleOp>(&op) ||
          isa<StoreOp>(&op) || isa<LoadOp>(&op))
        continue;

      builder.setInsertionPointToEnd(newEntry);
      if (auto getRow = dyn_cast<GetRowOp>(&op)) {
        auto call = builder.create<func::CallOp>(getRow.getLoc(),
                                                 TypeRange{ptrType},
                                                 pudGetRowSym,
                                                 ValueRange{});
        mapping.map(getRow.getResult(), call->getResult(0));
        continue;
      }

      if (auto ap = dyn_cast<APOp>(&op)) {
        Value addr = ap.getAddr();
        Value addrPtr = mapping.lookupOrDefault(addr);
        builder.create<func::CallOp>(ap.getLoc(),
                                     TypeRange{},
                                     pudApSym,
                                     ValueRange{addrPtr});
        continue;
      }

      if (auto aap = dyn_cast<AAPOp>(&op)) {
        Value src = aap.getSrcAddr();
        Value dst = aap.getDstAddr();
        Value srcPtr = mapping.lookupOrDefault(src);
        Value dstPtr = mapping.lookupOrDefault(dst);
        builder.create<func::CallOp>(aap.getLoc(),
                                     TypeRange{},
                                     pudAapSym,
                                     ValueRange{srcPtr, dstPtr});
        continue;
      }

      if (auto ret = dyn_cast<func::ReturnOp>(&op)) {
        builder.create<func::ReturnOp>(ret.getLoc());
        continue;
      }

      builder.clone(op, mapping);

    }
    oldEntry.erase();
  }
};

} // namespace

std::unique_ptr<Pass> pud::createConvertPuDToFuncPass() {
  return std::make_unique<ConvertPuDToFunc>();
}
