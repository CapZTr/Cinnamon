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

  let arguments = (ins
      AnyTypeOf<[
        BitwiseRowType,
        ControlRowType,
        DataRowType
      ]>:$srcAddr,
      AnyTypeOf<[
        BitwiseRowType,
        ControlRowType,
        DataRowType
      ]>:$dstAddr);

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  // let hasVerifier = 1;
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{}];

  let arguments = (ins
      AnyTypeOf<[
        BitwiseRowType,
        ControlRowType,
        DataRowType
      ]>:$addr);

  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr))";
  // let hasVerifier = 1;
}

def StoreOp : PuD_Op<"store", [Pure]> {
  let summary = "Store a slice into a DRAM subarray";
  let description = [{}];

  let arguments = (ins DataRowType:$firstRow, I64:$lineNum);

  let assemblyFormat = "$firstRow `,` $lineNum attr-dict `:` qualified(type($firstRow)) `,` type($lineNum)";
}

def LoadOp : PuD_Op<"load", [Pure]> {
  let summary = "Load a slice from a DRAM subarray";
  let description = [{}];

  let arguments = (ins DataRowType:$firstRow, I64:$lineNum);

  let assemblyFormat = "$firstRow `,` $lineNum attr-dict `:` qualified(type($firstRow)) `,` type($lineNum)";
}

#endif
