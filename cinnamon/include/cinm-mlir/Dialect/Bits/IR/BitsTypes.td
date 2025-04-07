//===- Types.td - Bits dialect types ------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_TYPES
#define BITS_TYPES

include "BitsBase.td"

def SliceType : Bits_Type<"Slice"> {
  let mnemonic = "slice";
  let summary = "Represents a bit-sliced data type.";

  let parameters = (ins "int64_t":$bitWidth, "int64_t":$vectorLength);
  let hasCustomAssemblyFormat = 1;
}

#endif
