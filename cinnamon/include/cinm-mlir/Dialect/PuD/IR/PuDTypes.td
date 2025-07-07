//===- Types.td - PuD dialect types ------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_TYPES
#define PUD_TYPES

include "PuDBase.td"

// def BitwiseRowType : PuD_Type<"BitwiseRowAddress"> {
//   let mnemonic = "row_bitwise";
//   let summary = "Represents a bitwise row within a DRAM subarray";

//   let parameters = (ins
//     "int64_t":$bankID,
//     "int64_t":$subarrayID,
//     "int64_t":$rowID
//   );
//   let hasCustomAssemblyFormat = 1;
// }

// def ControlRowType : PuD_Type<"ControlRowAddress"> {
//   let mnemonic = "row_control";
//   let summary = "Represents a control row within a DRAM subarray";

//   let parameters = (ins
//     "int64_t":$bankID,
//     "int64_t":$subarrayID,
//     "bool":$rowID
//   );
//   let hasCustomAssemblyFormat = 1;
// }

// def DataRowType : PuD_Type<"DataRowAddress"> {
//   let mnemonic = "row_data";
//   let summary = "Represents a data row within a DRAM subarray";

//   let parameters = (ins
//     "int64_t":$bankID,
//     "int64_t":$subarrayID,
//     "int64_t":$rowID
//   );
//   let hasCustomAssemblyFormat = 1;
// }

def BitwiseRowType : PuD_Type<"BitwiseRowAddress"> {
  let mnemonic = "row_bitwise";
  let summary = "Represents a bitwise row within a DRAM subarray";
}

def ControlRowType : PuD_Type<"ControlRowAddress"> {
  let mnemonic = "row_control";
  let summary = "Represents a control row within a DRAM subarray";
}

def DataRowType : PuD_Type<"DataRowAddress"> {
  let mnemonic = "row_data";
  let summary = "Represents a data row within a DRAM subarray";
}

#endif
