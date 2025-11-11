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

#endif // PUD_CONVERSION_PASSES
