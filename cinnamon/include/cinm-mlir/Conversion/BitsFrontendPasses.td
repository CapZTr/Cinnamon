//===- Passes.td - BitsFrontend passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits frontend conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_FRONTEND_CONVERSION_PASSES
#define BITS_FRONTEND_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertArithToBits : Pass<"convert-arith-to-bits", operation="func::FuncOp"> {
  let summary = "Convert Arith dialect to Bits Dialect";
  let constructor = "mlir::bits_frontend::createConvertArithToBitsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def ConvertLinalgToBits : Pass<"convert-linalg-to-bits", operation="func::FuncOp"> {
  let summary = "Convert Linalg dialect to Bits Dialect";
  let constructor = "mlir::bits_frontend::createConvertLinalgToBitsPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // BITS_FRONTEND_CONVERSION_PASSES
