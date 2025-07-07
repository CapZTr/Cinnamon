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

  let arguments = (ins RowType:$srcAddr, RowType:$dstAddr);

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{}];

  let arguments = (ins RowType:$addr);

  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr))";
}

def StoreOp : PuD_Op<"store", [Pure]> {
  let summary = "Store a slice into a DRAM subarray";
  let description = [{}];

  let arguments = (ins SliceType:$slice, RowType:$firstRow);

  let assemblyFormat = "$slice `,` $firstRow attr-dict `:` qualified(type($slice)) `,` qualified(type($firstRow))";
}

def LoadOp : PuD_Op<"load", [Pure]> {
  let summary = "Load a slice from a DRAM subarray";
  let description = [{}];

  let arguments = (ins RowType:$firstRow, I64:$rowNum);
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$firstRow `,` $rowNum attr-dict `:` qualified(type($firstRow)) `,` type($rowNum) `->` qualified(type($slice))";
}

def GetRowOp : PuD_Op<"get_row", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    I64:$bankID,
    I64:$subarrayID,
    I64:$rowID
  );
  let results = (outs RowType:$addr);

  let assemblyFormat = "$bankID `,` $subarrayID `,` $rowID attr-dict `:` type($bankID) `,` type($subarrayID) `,` type($rowID) `->` qualified(type($addr))";
}

#endif
