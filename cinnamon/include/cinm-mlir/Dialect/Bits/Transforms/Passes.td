//===- Passes.td - Bits dialect passes ---------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_TRANSFORM_PASSES
#define BITS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def BitsOptmiseMappingPass : Pass<"bits-opt-mapping", "func::FuncOp"> {
  let summary = "Generate optimal mapping using integer linear programming.";
  let description = [{}];
  let dependentDialects = [];
}

#endif // BITS_TRANSFORM_PASSES
