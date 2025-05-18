//===- Types.td - PuD dialect types ------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_TYPES
#define PUD_TYPES

include "PuDAttributes.td"
include "PuDBase.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

def RowAddressType : PuD_Type<"RowAddress"> {
  let mnemonic = "row_addr";
  let summary = "Represents a row in a DRAM subarray";

  let parameters = (ins
      PuDRowGroupAttr:$group,
      AttrParameter<"::mlir::IntegerAttr", "I64Attr", "::mlir::IntegerAttr">:$index
  );
  let hasCustomAssemblyFormat = 1;
}

#endif
