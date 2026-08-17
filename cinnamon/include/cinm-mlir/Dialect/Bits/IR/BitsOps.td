//===- Ops.td - Bits dialect ops ---------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_OPS
#define BITS_OPS

include "BitsBase.td"
include "BitsTypes.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def TransposeOp : Bits_Op<"transpose", [Pure]> {
  let summary = "Transpose 1-D tensor to Slice type, or 2-D tansor to Cube type";
  let description = [{}];

  let arguments = (ins TensorOf<[AnyInteger, AnyFloat]>:$input);
  let results = (outs AnyTypeOf<[SliceType, CubeType]>:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` "
      "qualified(type($output))";
  let hasVerifier = 1;
}

def AssembleOp : Bits_Op<"assemble", [Pure]> {
  let summary = "Assemble a bit-slice to 1-D tensor, or a bit-cube to 2-D tensor";
  let description = [{}];

  let arguments = (ins AnyTypeOf<[SliceType, CubeType, TensorOf<[SliceType]>]>:$input);
  let results = (outs TensorOf<[AnyInteger, AnyFloat]>:$output);

  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` "
      "type($output)";
  let hasVerifier = 1;
}

def ExtractSliceOp : Bits_Op<"extract_slice", [Pure]> {
  let summary = "Extract one slice with the given layer from a cube";
  let description = [{}];

  let arguments = (ins CubeType:$cube, I64:$layer);
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$cube `,` $layer attr-dict `:` "
      "qualified(type($cube)) `,` type($layer) `->` "
      "qualified(type($slice))";
  let hasVerifier = 1;
}

def ExtractSubSliceOp : Bits_Op<"extract_sub_slice", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins SliceType:$slice, I64:$rowNum);
  let results = (outs SliceType:$resultSlice);

  let assemblyFormat = "$slice `,` $rowNum attr-dict `:` "
      "qualified(type($slice)) `,` type($rowNum) `->` qualified(type($resultSlice))";
}

def ExtractRowOp : Bits_Op<"extract_row", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins SliceType:$slice, I64:$rowIdx);
  let results = (outs BitRowType:$row);

  let assemblyFormat = "$slice `,` $rowIdx attr-dict `:` "
      "qualified(type($slice)) `,` type($rowIdx) `->` qualified(type($row))";
}

def InsertRowOp : Bits_Op<"insert_row", [Pure]> {
  let arguments = (ins
    SliceType:$slice,
    BitRowType:$row,
    I64:$index
  );
  let results = (outs SliceType:$resultSlice);

  let assemblyFormat = "$slice `,` $row `,` $index attr-dict `:` "
      "qualified(type($slice)) `,` qualified(type($row)) `,` type($index) "
      "`->` qualified(type($resultSlice))";
}

def InsertSliceOp : Bits_Op<"insert_slice", [Pure]> {
  let summary = "Insert a slice to the given layer of a cube";
  let description = [{}];

  let arguments = (ins CubeType:$cube, SliceType:$slice, I64:$layer);
  let results = (outs CubeType:$resultCube);

  let assemblyFormat = "$cube `,` $slice `,` $layer attr-dict `:` "
      "qualified(type($cube)) `,` qualified(type($slice)) `,` type($layer) "
      "`->` qualified(type($resultCube))";
  let hasVerifier = 1;
}

def ExtractAndReplicateColumnOp : Bits_Op<"extract_and_replicate", [Pure]> {
  let summary = "Extract one column with the given index from a slice and replicate given time as a new slice";
  let description = [{}];

  let arguments = (ins SliceType:$slice, I64:$index, I64:$time);
  let results = (outs SliceType:$resultSlice);

  let assemblyFormat = "$slice `,` $index `,` $time attr-dict `:`"
      "qualified(type($slice)) `,` type($index) `,` type($time) `->` qualified(type($resultSlice))";
}

def ExtensionIOp : Bits_Op<"ext_i", [Pure]> {
  let summary = "Widen a slice by rows it does not have to store";
  let description = [{
    The rows above the operand's own width are a row the substrate already
    holds: the reserved zero row for an unsigned extension, the operand's own
    top row for a signed one. `isSigned` says which, so that the expansion that
    consumes the extended slice knows which row to read for those bits. Nothing
    is computed here and nothing is copied, which is why an extension is priced
    at zero -- but that only holds because whatever reads it reads a row that is
    already there, so an `ext_i` that survives to row level is an error rather
    than a free op.
  }];

  let arguments = (ins
    SliceType:$slice,
    I64:$rowNumToExt,
    UnitAttr:$isSigned
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$slice `,` $rowNumToExt attr-dict `:` "
      "qualified(type($slice)) `,` type($rowNumToExt) `->` "
      "qualified(type($result))";
  let hasVerifier = 1;
}

def SplitSliceVerticallyOp : Bits_Op<"split_vertically", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$slice,
    I64:$columnNum
  );
  let results = (outs
    SliceType:$first,
    SliceType:$second
  );

  let assemblyFormat = "$slice `,` $columnNum attr-dict `:` "
      "qualified(type($slice)) `,` type($columnNum) `->` "
      "qualified(type($first)) `,` qualified(type($second))";
  let hasVerifier = 1;
}

def MergeSliceVerticallyOp : Bits_Op<"merge_vertically", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$first,
    SliceType:$second
  );
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$first `,` $second attr-dict `:` "
      "qualified(type($first)) `,` qualified(type($second)) `->` "
      "qualified(type($slice))";
  let hasVerifier = 1;
}

def AddIOp : Bits_Op<"add_i", [Pure]> {
  let summary = "Bit-serial addition of two slice operands in integer";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
      "qualified(type($lhs)) `,` qualified(type($rhs))"
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def SubIOp : Bits_Op<"sub_i", [Pure]> {
  let summary = "Bit-serial subtraction of two slice operands in integer";
  let description = [{
    Two's complement difference, truncated to the operand width like
    `bits.add_i`: `lhs - rhs = lhs + ~rhs + 1`.

    On this substrate that is the SAME full-adder chain, with the carry
    initialised to one instead of zero and the subtrahend read complemented.
    The complement is not free, because the adder reads the subtrahend's row
    TWICE per bit (once into the majority that forms the carry, once into the
    sum), so it has to be materialised: SIMDRAM stages it through a dual-contact
    cell for one extra AAP per bit (**9*N + 1**, against 8*N + 1 for an add),
    PRADA for one extra single-command NOT per bit (**7*N + 1**, against
    6*N + 1). Subtraction is therefore the one arithmetic op where PRADA's
    advantage over Ambit is larger than it is for addition.

    Exists for squared-distance kernels ((x-y)^2 summed over a feature
    dimension), where the subtraction is unavoidable and folding it into the
    multiply is not possible.
  }];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
      "qualified(type($lhs)) `,` qualified(type($rhs))"
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def AddIFullOp : Bits_Op<"add_i_full", [Pure]> {
  let summary = "Widening bit-serial addition: N + N -> N+1 bits";
  let description = [{
    Unlike `bits.add_i`, which truncates to the operand width, this keeps the
    carry-out as an extra result row. That is what a reduction tree wants: two
    k-bit partial counts add to a (k+1)-bit one with no possibility of
    overflow, and without paying to pre-extend both operands to k+1 bits first.

    Cost is 8*N + 2 AAP: one carry initialisation, N full-adder bit steps at
    8 AAP each, and one final store of the carry-out row (the carry lives in a
    dual-contact row that the next addition's carry-init would clobber, so it
    must be copied into the result).
  }];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:`"
      "qualified(type($lhs)) `,` qualified(type($rhs))"
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def RowAddOp : Bits_Op<"row_add", [Pure]> {
  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs,
    BitRowType:$cin
  );
  let results = (outs
    BitRowType:$sum,
    BitRowType:$cout
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $cin attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "qualified(type($cin)) `->` qualified(type($sum)) `,` "
      "qualified(type($cout))";
}

def RowSubOp : Bits_Op<"row_sub", [Pure]> {
  let summary = "One bit position of a bit-serial subtraction";
  let description = [{
    The row-level form of `bits.sub_i`, and the counterpart of `bits.row_add`:
    difference and borrow-out from a minuend row, a subtrahend row and a
    borrow-in row.

    It is its own op rather than a `bits.not` on the subtrahend followed by a
    `bits.row_add` because that is not what the substrate does. Complementing
    the subtrahend is not a separate operation there -- it is which contact of a
    dual-contact cell the adder reads -- so spelling it as not-then-add would
    charge two commands for something that costs one, and the one it costs is
    already inside this op's nine.

    Like `bits.row_add`, whether the borrow stays resident in a dual-contact cell
    or has to be read from and written back to a data row is a property of the
    expansion that built it, recorded in the `bits.carry` attribute.
  }];

  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs,
    BitRowType:$bin
  );
  let results = (outs
    BitRowType:$diff,
    BitRowType:$bout
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $bin attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "qualified(type($bin)) `->` qualified(type($diff)) `,` "
      "qualified(type($bout))";
}

def MulIOp : Bits_Op<"mul_i", [Pure]> {
  let summary = "Bit-serial multiplication of two slice operands "
      "in integer";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def MulFOp : Bits_Op<"mul_f", [Pure]> {
  let summary = "Bit-serial multiplication of two slice operands "
      "in floating point";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def AndOp : Bits_Op<"and", [Pure]> {
  let summary = "Bit-serial AND of two slice operands";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def RowAndOp : Bits_Op<"row_and", [Pure]> {
  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "`->` qualified(type($result))";
}

def OrOp : Bits_Op<"or", [Pure]> {
  let summary = "Bit-serial OR of two slice operands";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def RowOrOp : Bits_Op<"row_or", [Pure]> {
  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "`->` qualified(type($result))";
}

def XOrOp : Bits_Op<"xor", [Pure]> {
  let summary = "Bit-serial XOR of two slice operands";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def XNOrOp : Bits_Op<"xnor", [Pure]> {
  let summary = "Bit-serial XNOR of two slice operands";
  let description = [{
    Bitwise XNOR, the equality predicate on each bit position. This is the
    core primitive of a binary neural network: with weights and activations
    encoded as +/-1 in one bit (0 -> -1, 1 -> +1), a binary multiply IS an
    XNOR, and the K-element dot product is `2 * popcount(xnor(x, w)) - K`.

    XNOR is a distinct op rather than `bits.xor` plus a complement because in
    a PuD substrate the two cost exactly the same -- both are three majority
    gates over the AMBIT control rows (7 row-op commands per bit), differing
    only in which constant row is read. Spelling XNOR as xor-then-not would
    charge for an inversion the hardware does not perform: `bits.not` is
    row-level, so a slice-level complement would cost an extra 2 commands per
    bit (9n instead of 7n), inflating our own reported runtime.
  }];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def RowXOrOp : Bits_Op<"row_xor", [Pure]> {
  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "`->` qualified(type($result))";
}

def RowXNorOp : Bits_Op<"row_xnor", [Pure]> {
  let summary = "Bit-row XNOR: the equality of two rows";
  let description = [{
    The row-level form of `bits.xnor`. It is its own op rather than a row XOR
    followed by a `bits.not` for the same reason the slice-level op is: on the
    substrate both are three majority gates over the control rows and differ
    only in which constant row is read, so spelling it as xor-then-complement
    would charge for an inversion the hardware never performs.
  }];

  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
}

def RangeScanOp : Bits_Op<"range_scan", [Pure]> {
  let summary = "BitWeaving/V BETWEEN range-scan predicate over a bit-sliced column";
  let description = [{
    Evaluates `lower < value < upper` over a whole bit-sliced database column
    in one op, producing a one-bit-per-value match mask.

    The input is a column in BitWeaving/V layout -- a `slice<L x N>` holding N
    values of L bits, bit-plane j of every value in row j -- which is exactly
    the layout a `bits.transpose` of a 1-D column tensor already produces. The
    result is a `slice<1 x N>` mask.

    Both comparison chains advance together, one majority gate each, using the
    single-bit recurrence (LSB to MSB):

        R (>= lower):  lower[j]==0 -> R = R OR  v[j] ;  ==1 -> R = R AND v[j]
        S (<= upper):  upper[j]==1 -> S = S OR ~v[j] ;  ==0 -> S = S AND ~v[j]

    `maj(sel, v, acc)` is OR when sel==1 and AND when sel==0, so one AP per
    chain per bit position advances it, and the dual-contact cell supplies
    `~v` for free. Cost is 4 commands per bit plus 5 of setup/finalize,
    independent of the constants -- every branch of the per-bit case analysis
    emits the same number of commands, only reading a different constant row.

    `lower` and `upper` are carried for IR self-description and for any future
    functional simulation; they do NOT affect the cost, for the reason above.

    The match COUNT is deliberately not part of this op. Counting a mask
    across lanes is a horizontal reduction, which is awkward on a PuD
    substrate; every published BitWeaving implementation this is modelled on
    does that final popcount on the host, and so do we.
  }];

  let arguments = (ins
    SliceType:$input,
    I64Attr:$lower,
    I64Attr:$upper
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$input attr-dict `:` "
      "qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def RowGreaterStepOp : Bits_Op<"row_greater_step", [Pure]> {
  let summary = "One bit of the comparison a bit-serial min or max starts with";
  let description = [{
    `g' = maj(lhs, !rhs, g)`, least significant bit upwards, so that after the
    most significant bit `g` says whether lhs > rhs. Equal bits carry the lower
    bits' verdict through untouched, and the first differing bit from the top
    settles it.

    The verdict stays resident in a compute row for the whole comparison, which
    is why the chain costs nothing to keep: it is read three times a bit and
    complemented none, so a compute row suits it where a dual-contact cell suits
    a carry.
  }];

  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs,
    BitRowType:$g
  );
  let results = (outs BitRowType:$gOut);

  let assemblyFormat = "$lhs `,` $rhs `,` $g attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "qualified(type($g)) `->` qualified(type($gOut))";
}

def RowSelectOp : Bits_Op<"row_select", [Pure]> {
  let summary = "One bit of a bit-serial select: g ? lhs : rhs";
  let description = [{
    The second half of a min or a max. `g` is the verdict
    `bits.row_greater_step` left, and this op reads both it and its complement,
    so it wants it in a dual-contact cell -- one copy, before the first bit, not
    one per bit.
  }];

  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs,
    BitRowType:$g
  );
  let results = (outs BitRowType:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $g attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "qualified(type($g)) `->` qualified(type($result))";
}

def RowScanStepOp : Bits_Op<"row_scan_step", [Pure]> {
  let summary = "One bit position of a BitWeaving range scan";
  let description = [{
    Advances both comparison chains of `bits.range_scan` by one bit plane of the
    scanned column.

        ge' = maj(ge,  value, !lower[j])     the ">= lower" chain
        le' = maj(le, !value,  upper[j])     the "<= upper" chain

    The bound bits are attributes, not operands, because they are compile-time
    constants that select which of four substrate sequences runs -- reading them
    from rows would make the scan pay to load constants it already knows. That
    is also why a range scan is expanded one bit position at a time rather than
    as a loop: consecutive bit planes need different sequences.

    Both chains stay resident in dual-contact cells for the whole scan, so
    neither ever reaches a data row; the only copies are the two that seed them.
  }];

  let arguments = (ins
    BitRowType:$value,
    BitRowType:$ge,
    BitRowType:$le,
    BoolAttr:$lowerBit,
    BoolAttr:$upperBit
  );
  let results = (outs
    BitRowType:$geOut,
    BitRowType:$leOut
  );

  let assemblyFormat = "$value `,` $ge `,` $le attr-dict `:` "
      "qualified(type($value)) `,` qualified(type($ge)) `,` "
      "qualified(type($le)) `->` qualified(type($geOut)) `,` "
      "qualified(type($leOut))";
}

def RowScanFoldOp : Bits_Op<"row_scan_fold", [Pure]> {
  let summary = "Combine a range scan's two chains into its match mask";
  let description = [{
    `mask = ge AND le`. Its own op rather than a `bits.maj` against a zero row
    because both operands are already resident in dual-contact cells: a general
    majority would be given three rows to stage, and this needs to stage one.
  }];

  let arguments = (ins
    BitRowType:$ge,
    BitRowType:$le
  );
  let results = (outs
    BitRowType:$mask
  );

  let assemblyFormat = "$ge `,` $le attr-dict `:` "
      "qualified(type($ge)) `,` qualified(type($le)) "
      "`->` qualified(type($mask))";
}

def MajOp : Bits_Op<"maj", [Pure]> {
  let summary = "Majority gate over three row operands";
  let description = [{}];

  let arguments = (ins
    BitRowType:$lhs,
    BitRowType:$rhs,
    BitRowType:$third
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs `,` $third attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` "
      "qualified(type($third)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def NotOp : Bits_Op<"not", [Pure]> {
  let summary = "Inverter operation on row operand";
  let description = [{}];

  let arguments = (ins
    BitRowType:$input
  );
  let results = (outs
    BitRowType:$result
  );

  let assemblyFormat = "$input attr-dict `:` "
      "qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def ReduceAndOp : Bits_Op<"and_red", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$input
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$input attr-dict `:` "
      "qualified(type($input)) `->` qualified(type($result))";
}

def ReduceOrOp : Bits_Op<"or_red", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$input
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$input attr-dict `:` "
      "qualified(type($input)) `->` qualified(type($result))";
}

def ReduceXOrOp : Bits_Op<"xor_red", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$input
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$input attr-dict `:` "
      "qualified(type($input)) `->` qualified(type($result))";
}

def MaxOp : Bits_Op<"max", [Pure]> {
  let summary = "Bit-serial max of two slice operands";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def MinOp : Bits_Op<"min", [Pure]> {
  let summary = "Bit-serial min of two slice operands";
  let description = [{}];

  let arguments = (ins
    SliceType:$lhs,
    SliceType:$rhs
  );
  let results = (outs
    SliceType:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def MuxOp : Bits_Op<"mux", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins SliceType:$lhs, SliceType:$rhs, SliceType:$s);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $s attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($s)) "
      "`->` qualified(type($result))";
}

def MatMulOp : Bits_Op<"matmul",  [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
      CubeType:$lhs,
      CubeType:$rhs
  );

  let results = (outs CubeType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def MatvecMulOp : Bits_Op<"matvecmul",  [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
      CubeType:$lhs,
      SliceType:$rhs
  );

  let results = (outs SliceType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` "
      "qualified(type($lhs)) `,` qualified(type($rhs)) "
      "`->` qualified(type($result))";
  let hasVerifier = 1;
}

def ShiftUpOp : Bits_Op<"shift_up", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins SliceType:$slice, I64:$numToShift);
  let results = (outs SliceType:$resultSlice);

  let assemblyFormat = "$slice `,` $numToShift attr-dict `:` "
      "qualified(type($slice)) `,` type($numToShift) `->` "
      "qualified(type($resultSlice))";
}

def CreateSliceOp : Bits_Op<"create_slice", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins I64:$bitWidth, I64:$vectorLength);
  let results = (outs SliceType:$slice);
  let assemblyFormat = "$bitWidth `,` $vectorLength attr-dict `:` "
      "type($bitWidth) `,` type($vectorLength) `->` qualified(type($slice))";
}

def CreateRowOp : Bits_Op<"create_row", [Pure]> {
  let arguments = (ins I64:$vectorLength);
  let results = (outs BitRowType:$row);
  let assemblyFormat = "$vectorLength attr-dict `:` "
      "type($vectorLength) `->` qualified(type($row))";
}

def CreateAllZeroRowOp : Bits_Op<"create_all_zero_row", [Pure]> {
  let arguments = (ins I64:$vectorLength);
  let results = (outs BitRowType:$row);
  let assemblyFormat = "$vectorLength attr-dict `:` "
      "type($vectorLength) `->` qualified(type($row))";
}

def CreateAllOneRowOp : Bits_Op<"create_all_one_row", [Pure]> {
  let summary = "A row of ones";
  let description = [{
    The counterpart of `bits.create_all_zero_row`. Both name a row the subarray
    holds at a constant, so neither costs anything to produce; what they cost is
    the command that reads them.

    A subtraction needs this one: two's complement is `lhs + ~rhs + 1`, so its
    borrow chain starts at one where an addition's carry starts at zero.
  }];

  let arguments = (ins I64:$vectorLength);
  let results = (outs BitRowType:$row);
  let assemblyFormat = "$vectorLength attr-dict `:` "
      "type($vectorLength) `->` qualified(type($row))";
}

def CreateAllOneSliceOp : Bits_Op<"create_all_one_slice", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins I64:$bitWidth, I64:$vectorLength);
  let results = (outs SliceType:$slice);
  let assemblyFormat = "$bitWidth `,` $vectorLength attr-dict `:` "
      "type($bitWidth) `,` type($vectorLength) `->` qualified(type($slice))";
}

def BroadcastRowOp : Bits_Op<"broadcast_row", [Pure]> {
  let summary = "Replicate one bit row into every row of a slice";
  let description = [{
    Presents a single bit row as a `bitWidth`-row slice so it can be the
    operand of a row-parallel slice op (e.g. `bits.and` for a bit-serial
    ReLU mask).

    This carries NO cost and never appears in a trace: the underlying DRAM
    row is simply used as an operand `bitWidth` times, which is exactly what
    the per-bit row-op expansion already does. It is a layout view, not a
    copy.
  }];

  let arguments = (ins BitRowType:$row, I64:$bitWidth);
  let results = (outs SliceType:$slice);
  let assemblyFormat = "$row `,` $bitWidth attr-dict `:` "
      "qualified(type($row)) `,` type($bitWidth) `->` qualified(type($slice))";
}

def CreateCubeOp : Bits_Op<"create_cube", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins I64:$bitWidth, I64:$vectorLength, I64:$height);
  let results = (outs CubeType:$cube);
  let assemblyFormat = "$bitWidth `,` $vectorLength `,` $height attr-dict `:` "
      "type($bitWidth) `,` type($vectorLength) `,` type($height) `->` qualified(type($cube))";
}

#endif
