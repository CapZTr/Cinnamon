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

  let arguments = (ins AnyTypeOf<[SliceType, CubeType]>:$input);
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
  let summary = "";
  let description = [{}];

  let arguments = (ins
    SliceType:$slice,
    I64:$rowNumToExt
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

def MatmulIOp : Bits_Op<"matmul_i",  [Pure]> {
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

def MatvecIOp : Bits_Op<"matvec_i",  [Pure]> {
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

def CreateSliceOp : Bits_Op<"create_slice", [Pure]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins I64:$bitWidth, I64:$vectorLength);
  let results = (outs SliceType:$slice);
  let assemblyFormat = "$bitWidth `,` $vectorLength attr-dict `:` "
      "type($bitWidth) `,` type($vectorLength) `->` qualified(type($slice))";
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
