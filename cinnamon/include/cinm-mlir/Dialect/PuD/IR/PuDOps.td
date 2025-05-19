//===- Ops.td - PuD dialect ops ----------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_OPS
#define PUD_OPS

include "PuDBase.td"
include "PuDTypes.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AAPOp : PuD_Op<"AAP", [Pure]> {
  let summary = "Activate ; Activate ; Precharge";
  let description = [{}];

  let arguments = (ins RowAddressType:$srcAddr, RowAddressType:$dstAddr);

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  let hasVerifier = 1;
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{}];

  let arguments = (ins RowAddressType:$addr);

  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr))";
  let hasVerifier = 1;
}

#endif
