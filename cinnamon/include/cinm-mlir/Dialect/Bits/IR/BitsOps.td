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
  let summary = "Transpose 1-D tensor of integers to Slice type";
  let description = [{}];

  let arguments = (ins TensorOf<[AnySignlessInteger]>:$input);
  let results = (outs SliceType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
  let hasVerifier = 1;
}

def AssembleOp : Bits_Op<"assemble", [Pure]> {
  let summary = "Assemble a bit-slice to 1-D tensor";
  let description = [{}];

  let arguments = (ins SliceType:$input);
  let results = (outs TensorOf<[AnySignlessInteger]>:$output);

  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
  let hasVerifier = 1;
}

def AddOp : Bits_Op<"add", [Pure, Commutative]> {
  let summary = "Bit-serial addition of two slice operands";
  let description = [{}];

  let arguments = (ins SliceType:$lhs, SliceType:$rhs);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";
  let hasVerifier = 1;
}






// ===== USELESS: To Be Modified =====

def AddBitOp : Bits_Op<"add_bit", [Pure]> {
  let summary = "Bitwise addition of two single bits";
  let description = [{}];

  let arguments = (ins BitType:$lhs, BitType:$rhs, BitType:$cin);

  let results = (outs BitType:$sum, BitType:$cout);

  let assemblyFormat = "$lhs `,` $rhs `,` $cin attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `,` qualified(type($cin)) `->` qualified(type($sum)) `,` qualified(type($cout))";
  let hasVerifier = 1;
}

def SubOp : Bits_Op<"sub", [Pure]> {
  let summary = "Bit-serial subtraction of two slice operands";
  let description = [{}];

  let arguments = (ins SliceType:$lhs, SliceType:$rhs);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def ExtractOp : Bits_Op<"extract", [Pure]> {
  let summary = "Extract a single bit from a bit-slice";
  let description = [{}];

  let arguments = (ins SliceType:$slice, I64:$bitIndex, I64:$vectorIndex);
  let results = (outs BitType:$bit);

  let assemblyFormat = "$slice `[` $bitIndex `,` $vectorIndex `]` attr-dict `:` qualified(type($slice)) `,` type($bitIndex) `,` type($vectorIndex) `->` qualified(type($bit))";
  let hasVerifier = 1;
}

def InsertOp : Bits_Op<"insert", [Pure]> {
  let summary = "Insert a bit into a bit-slice at a position encoded in BitType";
  let description = [{}];

  let arguments = (ins SliceType:$slice, BitType:$bit, I64:$bitIndex, I64:$vectorIndex);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$bit `->` $slice `[` $bitIndex `,` $vectorIndex `]` attr-dict `:` qualified(type($bit)) `,` qualified(type($slice)) `,` type($bitIndex) `,` type($vectorIndex) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def CreateSliceOp : Bits_Op<"create_slice", [Pure]> {
  let summary = "Create a zero-valued Slice";
  let description = [{}];

  let arguments = (ins I64:$bitWidth, I64:$vectorLength);
  let results = (outs SliceType:$slice);

  let assemblyFormat = "$bitWidth `,` $vectorLength attr-dict `:` type($bitWidth) `,` type($vectorLength) `->` qualified(type($slice))";
}

def CreateBitOp : Bits_Op<"create_bit", [Pure]> {
  let summary = "Create a zero-valued Bit";
  let description = [{}];

  let results = (outs BitType:$bit);

  let assemblyFormat = "attr-dict `:` qualified(type($bit))";
}

#endif
