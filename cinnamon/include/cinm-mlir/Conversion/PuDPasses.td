//===- Passes.td - PuD dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_CONVERSION_PASSES
#define PUD_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPuDToFunc : Pass<"convert-pud-to-func", operation="func::FuncOp"> {
  let summary = "Convert PuD dialect to Func Dialect";
  let constructor = "mlir::pud::createConvertPuDToFuncPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::pud::PuDDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertPuDRowsToFunc
    : Pass<"convert-pud-rows-to-func", operation="func::FuncOp"> {
  let summary = "Lower PuD row ops to guest runtime calls";
  let description = [{
    Replaces each row op with a call to the guest stub that issues it, and each
    row address with a call into the runtime that owns the row allocation. The
    loop structure is left alone, so the standard scf -> cf -> llvm pipeline
    takes it the rest of the way.
  }];
  let constructor = "mlir::pud::createConvertPuDRowsToFuncPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::pud::PuDDialect"
  ];
}

#endif // PUD_CONVERSION_PASSES
