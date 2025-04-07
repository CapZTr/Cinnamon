//===- Ops.td - Bits dialect ops ----------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_OPS
#define BITS_OPS


include "cinm-mlir/Dialect/Bits/IR/BitsBase.td"
include "cinm-mlir/Dialect/Bits/IR/BitsTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"


def TransposeOp : Bits_Op<"transpose", [Pure]> {
  let summary = "Transpose 1-D tensor of integers to Slice type";
  let description = [{}];

  let arguments = (ins TensorOf<[AnySignlessInteger]>:$input);
  let results = (outs SliceType:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def AddOp : Bits_Op<"add", [Pure, Commutative]> {
  let summary = "Bit-serial addition of two slice operands.";
  let description = [{}];

  let arguments = (ins SliceType:$lhs, SliceType:$rhs);
  let results = (outs SliceType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
  let hasVerifier = 1;
}

#endif
