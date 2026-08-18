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
  let options = [
    Option<"convLaneDim", "conv-lane-dim", "std::string", /*default=*/"\"auto\"",
           "Which convolution dimension becomes the bit-serial SIMD lane "
           "dimension (the GEMM's M). 'oc' = output channels (M=F, outer "
           "trip=P*Q); 'spatial' = output pixels (M=P*Q, outer trip=F); "
           "'auto' takes the wider of the two. This is a first-order mapping "
           "knob: the two layouts perform identical arithmetic but can differ "
           "by >4x. 'auto' is a shape rule, not a second copy of the mapper's "
           "cost model -- the lane axis lies along a bank row, whatever it "
           "leaves empty gets filled by packing the reduction, and every "
           "doubling of that packing costs a level of intra-bank folding that "
           "computes nothing. It reads the operand and the row width and "
           "nothing about banks or latencies, so it cannot drift from "
           "planLeafForRegions the way an estimate would. Sweep both "
           "explicitly to compare layouts.">,
    Option<"wideAdd", "wide-add", "bool", /*default=*/"true",
           "Whether the target substrate offers a WIDENING add (k + k -> k+1, "
           "`bits.add_i_full`, 8k + 2 AAP). Only the width-growing adder trees "
           "-- popcount and the binarised dot product -- can use one. With "
           "false they fall back to what a substrate without it must do: "
           "zero-extend both operands to k+1 first and use an equal-width "
           "`bits.add_i`, costing 8(k+1) + 1 per node. The arithmetic and the "
           "result are identical either way, so this isolates the cost of the "
           "capability itself. Note this is the FRONTEND half of that ablation "
           "and is independent of bits-opt-mapping's own `wide-add`, which "
           "instead promotes plain `bits.add_i` in the non-tree kernels; the "
           "figure sets the two together.">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // BITS_FRONTEND_CONVERSION_PASSES
