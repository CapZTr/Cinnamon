//===- Ops.td - PuD dialect ops ----------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_OPS
#define PUD_OPS

include "PuDBase.td"
include "PuDTypes.td"

include "cinm-mlir/Dialect/Bits/IR/BitsTypes.td"

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
        DataRowType
      ]>:$dstAddr);

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  // let hasVerifier = 1;
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{}];

  // let arguments = (ins
  //     AnyTypeOf<[
  //       BitwiseRowType,
  //       ControlRowType,
  //       DataRowType
  //     ]>:$addr);
  let arguments = (ins BitwiseRowType:$addr);

  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr))";
  // let hasVerifier = 1;
}

def StoreOp : PuD_Op<"store", [Pure]> {
  let summary = "Store a slice into a DRAM subarray";
  let description = [{}];

  let arguments = (ins SliceType:$slice);
  let results = (outs DataRowType:$firstRow);

  let assemblyFormat = "$slice attr-dict `:` qualified(type($slice)) `->` qualified(type($firstRow))";
}

def LoadOp : PuD_Op<"load", [Pure]> {
  let summary = "Load a slice from a DRAM subarray";
  let description = [{}];

  let arguments = (ins DataRowType:$firstRow);
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$firstRow attr-dict `:` qualified(type($firstRow)) `->` qualified(type($slice))";
}

def CalculateAddressOp : PuD_Op<"calculate_addr", [Pure]> {
  let summary = "Calculate address from a given one and an offset";
  let description = [{}];

  let arguments = (ins DataRowType:$base, I64Attr:$offset);
  let results = (outs DataRowType:$addr);

  let assemblyFormat = "$base `,` $offset attr-dict `:` qualified(type($base)) `->` qualified(type($addr))";
}

def GetRowOp : PuD_Op<"get_row", [Pure]> {
  let summary = "";
  let description = [{}];

  let results = (outs
      AnyTypeOf<[
        BitwiseRowType,
        ControlRowType,
        DataRowType
      ]>:$addr
  );

  let assemblyFormat = "attr-dict `:` qualified(type($addr))";
}

#endif
