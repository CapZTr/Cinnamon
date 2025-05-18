//===- Attributes.td - PuD dialect attributes --------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_ATTRIBUTES
#define PUD_ATTRIBUTES

include "PuDBase.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"

def PuDRowGroupAttr : I64EnumAttr<
    "RowGroup", "",
    [
      I64EnumAttrCase<"B", 0, "B">,
      I64EnumAttrCase<"C", 1, "C">,
      I64EnumAttrCase<"D", 2, "D">
    ]> {
  let cppNamespace = "mlir::pud";
}

#endif
