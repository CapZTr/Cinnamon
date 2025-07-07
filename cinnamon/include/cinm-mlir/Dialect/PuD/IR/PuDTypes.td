//===- Types.td - PuD dialect types ------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_TYPES
#define PUD_TYPES

include "PuDBase.td"

def RowType : PuD_Type<"Row"> {
  let mnemonic = "row";
  let summary = "Represents a row within a DRAM subarray";

  let parameters = (ins "int64_t":$group);
  let hasCustomAssemblyFormat = 1;
}

#endif
