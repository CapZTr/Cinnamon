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

  let assemblyFormat = "$slice `,` $columnNum attr-dict `:` qualified(type($slice)) `,` type($columnNum) `->` qualified(type($first)) `,` qualified(type($second))";
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

  let assemblyFormat = "$first `,` $second attr-dict `:` qualified(type($first)) `,` qualified(type($second)) `->` qualified(type($slice))";
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

def SubOp : Bits_Op<"sub", [Pure]> {
  let summary = "Bit-serial subtraction of two slice operands";
  let description = [{}];

  let arguments = (ins SliceType:$lhs, SliceType:$rhs);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($lhs)) `,` qualified(type($rhs)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

#endif
