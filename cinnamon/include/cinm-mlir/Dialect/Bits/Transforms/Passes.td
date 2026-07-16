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
  let options = [
    Option<"numBanks", "num-banks", "int64_t", /*default=*/"32",
           "Total number of PuD banks to map onto (power of two, <= 128)">,
    Option<"memType", "mem-type", "std::string", /*default=*/"\"ddr\"",
           "Target memory: 'ddr' (32 banks/channel; inter-bank row-copy cost "
           "19/10 AAP-units per bit intra-/cross-channel) or 'hbm' (16 "
           "banks/channel; 6/3)">,
  ];
}

def BitsCubeToSlicePass : Pass<"bits-cube-to-slice", "func::FuncOp"> {
  let summary = "Transform cube computation with slice computaton with loop structure.";
  let description = [{}];
  let dependentDialects = [];
}

def BitsPatternApplyPass : Pass<"bits-pattern-apply", "func::FuncOp"> {

}

#endif // BITS_TRANSFORM_PASSES
