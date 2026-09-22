// Binarised fully-connected layer, 4096 -> 8192, spelled over UNPACKED i1.
//
// This is the kernel benchmark/bnn/i64 could not express.  That one packs 64
// binary values into an i64 and reduces them with math.ctpop, which has no
// multi-word form -- so it measures ONE K=64 tile and its own header warns that
// a real layer is 16x it plus the cross-tile chaining.  Spelling the operands as
// i1 tensors removes the cap entirely: the reduction dimension is just a tensor
// dimension, so K = 4096 is a whole layer and nothing has to be extrapolated.
//
//   8192 output neurons  (the SIMD lane dimension, exactly one 8192-lane row)
//   4096 inputs          (the reduction dimension)
//      1 activation vector (batch 1)
//
// With +/-1 binarised into one bit (0 -> -1, 1 -> +1) a binary multiply IS an
// XNOR, and the dot product is 2*popcount(xnor(w, a)) - K.  This kernel computes
// the popcount(xnor(...)) core; the affine 2p - K tail is one add per output
// against the ~102k row ops the core costs, so it is left to the host exactly as
// in the i64 kernel.
//
// MLIR has no xnor op, so the body uses the double-xori (a ^ b) ^ true that
// every binary-network kernel in the wild is written with, then zero-extends the
// 1-bit product into the accumulator.  Note that the body describes a LINEAR
// accumulation; the lowering deliberately emits a width-growing adder tree
// instead, which is legal because the summands are 1-bit and addition is
// associative.  Taking the body literally would cost K*(8*32+1) = 1,052,672 AAP
// against 102,286 for the tree -- see ConvertLinalgBinaryDotToBits.
//
// The i32 accumulator is wider than needed (K = 4096 tops out at 13 bits) but
// costs nothing: the tree grows to its natural 13 bits and the final widening is
// a bits.ext_i, which is a reserved constant row, not arithmetic.

#w_map = affine_map<(d0, d1, d2) -> (d0, d2)>   // weights[neuron][input]
#a_map = affine_map<(d0, d1, d2) -> (d1, d2)>   // acts[batch][input]
#o_map = affine_map<(d0, d1, d2) -> (d0, d1)>   // out[neuron][batch]

func.func @bnn_dot_8192_4096(%w: tensor<8192x4096xi1>,
                             %a: tensor<1x4096xi1>) -> tensor<8192x1xi32> {
  %zero = arith.constant 0 : i32
  %true = arith.constant true
  %init = tensor.empty() : tensor<8192x1xi32>
  %acc0 = linalg.fill ins(%zero : i32)
                      outs(%init : tensor<8192x1xi32>) -> tensor<8192x1xi32>
  %r = linalg.generic {
    indexing_maps = [#w_map, #a_map, #o_map],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%w, %a : tensor<8192x4096xi1>, tensor<1x4096xi1>)
    outs(%acc0 : tensor<8192x1xi32>) {
  ^bb0(%wb: i1, %ab: i1, %acc: i32):
    %xor  = arith.xori %wb, %ab : i1
    %xnor = arith.xori %xor, %true : i1
    %e    = arith.extui %xnor : i1 to i32
    %s    = arith.addi %acc, %e : i32
    linalg.yield %s : i32
  } -> tensor<8192x1xi32>
  return %r : tensor<8192x1xi32>
}
