//===- Passes.td - Bits dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_TRANSFORM_PASSES
#define BITS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def BitsLowerArithPass: Pass<"bits-lower-arith", operation="mlir::func::FuncOp"> {
  let summary = "Lower arithmetic operations on slice-level to bit-level";
  let description = [{}];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // BITS_TRANSFORM_PASSES
