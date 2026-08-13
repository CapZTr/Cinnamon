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
    Option<"costModel", "cost-model", "std::string", /*default=*/"\"exact\"",
           "Cost model the mapper optimises against. 'exact': the calibrated "
           "per-op row-op counts of the target substrate plus location-aware "
           "row-copy pricing (what the simulator actually replays). 'proxy': "
           "one time unit per bit-plane row for EVERY op, compute and data "
           "movement alike, with no per-op and no per-location calibration -- "
           "the uncalibrated model an ablation compares against. Only the "
           "mapping DECISIONS change; the emitted row-op sequence, and hence "
           "the work the simulator performs, is identical either way.">,
    Option<"proxyBlind", "proxy-blind", "std::string", /*default=*/"\"\"",
           "With cost-model=proxy, which modelling features the proxy lacks, "
           "as a comma list (default: all of them, 'width,location,pack'). "
           "'oplevel' = one unit per bit-plane row instead of the op's real "
           "expansion count; 'width' = one unit per op and per copy, bitwidth "
           "ignored entirely (implies oplevel); 'location' = every inter-bank "
           "copy priced alike, hence a flat stride-halving reduce tree; 'pack' "
           "= every loop iteration priced as its own full-row wave, ignoring "
           "that a bank row holds many SIMD lanes. Listing them separately is "
           "what lets an ablation attribute a slowdown to a named feature.">,
    Option<"backend", "backend", "std::string", /*default=*/"\"simdram\"",
           "PuD substrate the exact costs are calibrated against: 'simdram' "
           "(Ambit AAP/AP) or 'prada' (sequential row activation: TRA / "
           "single-command NOT / 5RA). The two differ per op and not by a "
           "constant factor -- PRADA's adder and multiplier are cheaper, its "
           "XNOR and range scan dearer -- so a schedule is only optimal for "
           "the substrate it was priced for. Must match the simulator's "
           "--backend.">,
    Option<"wideAdd", "wide-add", "bool", /*default=*/"false",
           "Emit every plain bits.add_i as a WIDENING add (trace kind "
           "OP_ADDI_WIDE): one extra row-op stores the carry-out, so the sum "
           "is one bit wider and cannot wrap. Both the mapping cost and the "
           "emitted record change together. NOTE the IR result type is not "
           "re-typed -- this measures the cost of keeping the carry, it does "
           "not re-plumb the dataflow to consume the wider value.">,
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
