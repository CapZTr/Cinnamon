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

def ConvertBitsRowsToPuD
    : Pass<"convert-bits-rows-to-pud", operation="func::FuncOp"> {
  let summary = "Lower row-level Bits ops to PuD row ops";
  let description = [{
    Turns each row-level Bits op into the row-op sequence the substrate actually
    executes, by instantiating the template RowOpLibrary compiled for it and
    binding the template's operands to concrete row slots.

    This runs after `bits-opt-mapping` and `bits-pattern-apply`: the first
    annotates the tile loop with the banks it was mapped onto, the second turns
    slice ops into loops over bit rows. Each row op is emitted once per bank in
    the mapped range, command by command -- the same order the trace player
    replays, which is what lets banks overlap their activations.

    Distinct from `convert-bits-to-pud`, which builds one majority-inverter
    graph for a whole function and does not scale past small kernels.
  }];
  let constructor = "mlir::bits::createConvertBitsRowsToPuDPass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bits::BitsDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::pud::PuDDialect"
  ];
  let options = [
    Option<"backend", "backend", "std::string", /*default=*/"\"simdram\"",
           "PuD substrate whose row-op templates are instantiated. Must match "
           "the backend the mapping was priced for.">,
    Option<"rowStride", "row-stride", "int64_t", /*default=*/"0",
           "Bytes from one bank's row to the next bank's, which is the row "
           "size of the substrate. The rows a kernel addresses are laid out at "
           "this stride, so this is what turns a row coordinate into an "
           "address; it travels with the IR for whatever realises them. Zero "
           "takes the machine's own -- 8 kB on DDR4, 1 kB on HBM -- which the "
           "mapping recorded; anything else asks what another row size would "
           "cost, and will not decode on the machine that was mapped for.">,
    Option<"rowsPerSubarray", "rows-per-subarray", "int64_t", /*default=*/"512",
           "Rows a subarray holds. A row op drives a whole subarray's sense "
           "amplifiers, so every row it names has to be in one -- the rows it "
           "reserves for computation included, and those are at the front of "
           "the space. That makes this the number of rows a kernel has to fit "
           "in, not a number it may exceed at a cost.">,
    Option<"operandPerWave", "operand-per-wave", "bool", /*default=*/"true",
           "Give every wave of a reduction its own rows of the operand it "
           "reads, rather than one set per pass that its waves share. On means "
           "the operand is resident and no two waves read the same values, "
           "which is what a kernel has to be to compute the right answer; it "
           "costs rows, and a kernel whose operand does not fit in a subarray "
           "is told so. Off shares a pass's rows between its waves, which fits "
           "anything but computes on the same tile several times over -- for a "
           "kernel too big to be resident, where what is wanted is the "
           "schedule's cost rather than its result.">
  ];
}

#endif // BITS_CONVERSION_PASSES
