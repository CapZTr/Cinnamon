//===- Passes.td - Bits dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_CONVERSION_PASSES
#define BITS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertBitsToPuD : Pass<"convert-bits-to-pud", operation="func::FuncOp"> {
  let summary = "Convert Bits dialect to PuD Dialect";
  let constructor = "mlir::bits::createConvertBitsToPuDPass()";
  let dependentDialects = [
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::pud::PuDDialect"
  ];
}

#endif // BITS_CONVERSION_PASSES
