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

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` "
      "qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  let hasVerifier = 1;
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{}];

  let arguments = (ins RowType:$addr);

  let assemblyFormat = "$addr attr-dict `:` qualified(type($addr))";
  let hasVerifier = 1;
}

def StoreOp : PuD_Op<"store", [Pure]> {
  let summary = "Store a slice into a DRAM subarray";
  let description = [{}];

  let arguments = (ins SliceType:$slice, RowType:$firstRow);

  let assemblyFormat = "$slice `,` $firstRow attr-dict `:` "
      "qualified(type($slice)) `,` qualified(type($firstRow))";
  let hasVerifier = 1;
}

def LoadOp : PuD_Op<"load", [Pure]> {
  let summary = "Load a slice from a DRAM subarray";
  let description = [{}];

  let arguments = (ins RowType:$firstRow, I64:$rowNum);
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$firstRow `,` $rowNum attr-dict `:` "
      "qualified(type($firstRow)) `,` type($rowNum) `->` "
      "qualified(type($slice))";
  let hasVerifier = 1;
}

def RowAddrOp : PuD_Op<"row_addr", [Pure]> {
  let summary = "Address of one bank's copy of a row slot";
  let description = [{
    A row slot is a logical row that exists once per bank: slots 0-17 are the
    control rows the guest runtime sets up (T0-T3, the dual-contact cells, their
    activation groups and the two constant rows), and slots 18 and up are the
    data pools the kernel computes in. `bank` selects which bank's copy this
    address refers to, so issuing the same slot across a bank range is what
    makes banks run a row op in parallel.

    This is the only way a row address enters the PuD dialect. It replaces
    `get_row`'s channel/rank/bank/subarray/row tuple, which described a
    coordinate the guest runtime has no way to ask for: under gem5's SE mode the
    program allocates row-aligned blocks and lets the address map decide where
    they land, so a slot index plus a bank is exactly as much as the program can
    actually name.
  }];

  let arguments = (ins I64:$slot, I64:$bank);
  let results = (outs RowType:$addr);

  let assemblyFormat = "$slot `,` $bank attr-dict `:` type($slot) `,` "
      "type($bank) `->` qualified(type($addr))";
}

def GetRowOp : PuD_Op<"get_row", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    I64:$channelID,
    I64:$rankID,
    I64:$bankID,
    I64:$subarrayID,
    I64:$rowID
  );
  let results = (outs RowType:$addr);

  let assemblyFormat = "$channelID `,` $rankID `,` $bankID `,` $subarrayID `,` "
      "$rowID attr-dict `:` type($channelID) `,` type($rankID) `,` "
      "type($bankID) `,` type($subarrayID) `,` type($rowID) `->` "
      "qualified(type($addr))";
}

#endif
