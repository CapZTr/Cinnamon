//===- Ops.td - PuD dialect ops ----------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_OPS
#define PUD_OPS

include "PuDBase.td"
include "PuDAttributes.td"
include "PuDTypes.td"

include "cinm-mlir/Dialect/Bits/IR/BitsTypes.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def AAPOp : PuD_Op<"AAP", [Pure]> {
  let summary = "Activate ; Activate ; Precharge";
  let description = [{
    Copies the source row into the destination, which may name a set of rows
    driven by one activation rather than a single row. Both operand staging and
    result write-back are this op; a compute step is an AP on a row set whose
    activation leaves the majority of its rows in all of them.
  }];

  let arguments = (ins
    RowType:$srcAddr,
    RowType:$dstAddr,
    PuD_BackendAttr:$backend
  );

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` "
      "qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  let hasVerifier = 1;
}

def RowCopyOp : PuD_Op<"row_copy", [Pure]> {
  let summary = "Copy a row into another bank's row";
  let description = [{
    Moves one row of data from where it is to where the next computation on it
    has to be, which is what a reduction across banks is made of: the banks a
    reduction was spread over cannot read each other's rows, so a value crosses
    before it is folded in.

    Unlike `pud.AAP`, the two rows need not be in the same subarray, bank or
    rank, and what it costs depends on how far apart they are: rows sharing
    sense amplifiers move through them, rows in different banks of one channel
    move over that channel's data bus, and rows in different channels have to be
    read out of one and written into the other. That is the substrate's business
    -- here it is one command naming two rows.
  }];

  let arguments = (ins
    RowType:$srcAddr,
    RowType:$dstAddr,
    PuD_BackendAttr:$backend
  );

  let assemblyFormat = "$srcAddr `,` $dstAddr attr-dict `:` "
      "qualified(type($srcAddr)) `,` qualified(type($dstAddr))";
  let hasVerifier = 1;
}

def APOp : PuD_Op<"AP", [Pure]> {
  let summary = "Activate ; Precharge";
  let description = [{
    Activates one row set and precharges. Where the set holds three rows the
    activation leaves their majority in every one of them, which is how a
    compute step is expressed; nothing is read out.
  }];

  let arguments = (ins RowType:$addr, PuD_BackendAttr:$backend);

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

def GetCtrlRowOp : PuD_Op<"get_ctrl_row", [Pure]> {
  let summary = "Address of a reserved control row";
  let description = [{
    Names one of the rows a subarray reserves for computation -- the compute
    rows, the dual-contact cells and their complements, the row sets a single
    activation drives, and the two constant rows. Which rows exist and which
    sets are wired together is a property of the substrate, so they are named
    rather than computed, and the name is what appears in the IR.

    Only the subarray has to be given: a control row is at the same place in
    every subarray of the substrate.
  }];

  let arguments = (ins
    I64:$channelID,
    I64:$rankID,
    I64:$bankID,
    I64:$subarrayID,
    PuD_SimdramCtrlRowAttr:$ctrlRow
  );
  let results = (outs RowType:$addr);

  let assemblyFormat = "$ctrlRow `,` $channelID `,` $rankID `,` $bankID `,` "
      "$subarrayID attr-dict `:` type($channelID) `,` type($rankID) `,` "
      "type($bankID) `,` type($subarrayID) `->` qualified(type($addr))";
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
