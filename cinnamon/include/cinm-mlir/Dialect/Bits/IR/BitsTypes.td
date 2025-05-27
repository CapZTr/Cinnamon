//===- Types.td - Bits dialect types -----------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_TYPES
#define BITS_TYPES

include "BitsBase.td"

def RowAddressType : Bits_Type<"RowAddress"> {
  let mnemonic = "row_address";
  let summary = "Represents a row within a DRAM subarray";

  let parameters = (ins
      "int64_t":$bankID,
      "int64_t":$subarrayID,
      "int64_t":$rowID
  );
  let hasCustomAssemblyFormat = 1;
}

def SliceType : Bits_Type<"Slice"> {
  let mnemonic = "slice";
  let summary = "Represents a bit-sliced data type";

  let parameters = (ins
      "int64_t":$bitWidth,
      "int64_t":$vectorLength,
      RowAddressType:$firstRowAddr
  );
  let hasCustomAssemblyFormat = 1;
}

// ===== To Be Modified =====

def BitType : Bits_Type<"Bit"> {
  let mnemonic = "bit";
  let summary = "Represents a single bit within a bit-slice";
}

#endif
