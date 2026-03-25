//===- Types.td - Bits dialect types -----------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_TYPES
#define BITS_TYPES

include "cinm-mlir/Dialect/Bits/IR/BitsBase.td"

def RowType : Bits_Type<"Row"> {
  let mnemonic = "row";
  let summary = "Represents a one-dimensional bit-sliced data type";

  let parameters = (ins
      "int64_t":$vectorLength);
  let hasCustomAssemblyFormat = 1;
}

def SliceType : Bits_Type<"Slice"> {
  let mnemonic = "slice";
  let summary = "Represents a two-dimensional bit-sliced data type";

  let parameters = (ins
      "int64_t":$bitWidth,
      "int64_t":$vectorLength);
  let hasCustomAssemblyFormat = 1;
}

def CubeType : Bits_Type<"Cube"> {
  let mnemonic = "cube";
  let summary = "Represents a three-dimensional bit-sliced data type";

  let parameters = (ins
      "int64_t":$bitWidth,
      "int64_t":$vectorLength,
      "int64_t":$height
  );
  let hasCustomAssemblyFormat = 1;
}

#endif
