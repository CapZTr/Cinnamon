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
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::pud::PuDDialect"
  ];
  let options = [
    Option<"unroll", "unroll", "bool", "false", "Unroll the loops">
  ];
}

#endif // BITS_CONVERSION_PASSES
