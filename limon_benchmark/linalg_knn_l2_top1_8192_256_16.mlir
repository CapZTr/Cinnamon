// k-nearest-neighbour search, k = 1, squared-L2 distance over quantised i8
// features.  Unlike the earlier top-1 kernel in this directory, this one
// actually computes the distance: sum_d (q[d] - b[d])^2.  That summation is
// the kernel -- the selection is a 0.04% tail (15 comparisons against 4096
// multiply-accumulates per query) -- and it is a reduction over the feature
// dimension, which is what makes this a mapping problem at all.
//
//   8192 queries (the SIMD lane dimension, exactly one 8192-lane bank row)
//    256 features per vector (the reduction dimension)
//     16 database vectors  --  ONE SHARD, see the disclosure below
//
// Widths are chosen so that nothing overflows anywhere, which costs more than
// the naive spelling and is the honest choice:
//   * the difference is computed in i16, because an i8 difference does not fit
//     in i8 (features in [-128,127] give differences in [-255,255]);
//   * the square is a WIDENING i16 x i16 -> i32 multiply, spelled as an extsi
//     pair feeding arith.muli, because (x-y)^2 reaches 65025 and needs 17 bits;
//   * the accumulator is i32: 256 features x 65025 < 2^31.
// Charging the full i32 accumulate (8*32+1 = 257 AAP per step) rather than
// truncating to i16 over-reports our own cost, which is the direction to err in.
//
// DISCLOSURE, same as every other kernel in this suite: 16 database vectors is
// ONE SHARD.  A million-vector index is 65536 invocations of this kernel plus a
// merge of their 65536 partial winners; the per-shard cost measured here is the
// right unit to multiply, but the trace must not be read as a whole search.
//
// The selection step is a separate linalg.generic on purpose: a fused
// distance-and-select op would hide that the two have completely different
// arithmetic (a deep i32 accumulation vs a 15-step unsigned minimum), and the
// emitter chains sibling loops through a function-level cursor anyway, so the
// two land in one trace with the second following the first.

#q_map    = affine_map<(d0, d1, d2) -> (d0, d2)>   // query[lane][feature]
#b_map    = affine_map<(d0, d1, d2) -> (d1, d2)>   // db[vector][feature]
#dist_map = affine_map<(d0, d1, d2) -> (d0, d1)>   // dist[lane][vector]
#in_map   = affine_map<(d0, d1) -> (d0, d1)>
#out_map  = affine_map<(d0, d1) -> (d0)>

func.func @knn_l2_top1(%queries: tensor<8192x256xi8>,
                       %database: tensor<16x256xi8>) -> tensor<8192xi32> {
  %zero = arith.constant 0 : i32
  %dist_init = tensor.empty() : tensor<8192x16xi32>
  %dist_zero = linalg.fill ins(%zero : i32)
                           outs(%dist_init : tensor<8192x16xi32>)
                           -> tensor<8192x16xi32>

  // Distance: sum over the feature dimension of the squared difference.
  %dist = linalg.generic {
    indexing_maps = [#q_map, #b_map, #dist_map],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%queries, %database : tensor<8192x256xi8>, tensor<16x256xi8>)
    outs(%dist_zero : tensor<8192x16xi32>) {
  ^bb0(%q: i8, %b: i8, %acc: i32):
    %q16 = arith.extsi %q : i8 to i16
    %b16 = arith.extsi %b : i8 to i16
    %diff = arith.subi %q16, %b16 : i16
    // Widening square: both operands extended to the result width, which is
    // how a widening multiply is spelled in linalg (named elementwise ops are
    // type-homogeneous, so there is no i16 x i16 -> i32 named form).
    %d32a = arith.extsi %diff : i16 to i32
    %d32b = arith.extsi %diff : i16 to i32
    %sq = arith.muli %d32a, %d32b : i32
    %sum = arith.addi %acc, %sq : i32
    linalg.yield %sum : i32
  } -> tensor<8192x16xi32>

  // Selection: the nearest of the 16 shard vectors, per query lane.  Unsigned
  // minimum is exact here -- squared distances are non-negative.
  %big = arith.constant 2147483647 : i32
  %best_init = tensor.empty() : tensor<8192xi32>
  %best_seed = linalg.fill ins(%big : i32)
                           outs(%best_init : tensor<8192xi32>)
                           -> tensor<8192xi32>
  %best = linalg.generic {
    indexing_maps = [#in_map, #out_map],
    iterator_types = ["parallel", "reduction"]
  } ins(%dist : tensor<8192x16xi32>)
    outs(%best_seed : tensor<8192xi32>) {
  ^bb0(%d: i32, %acc: i32):
    %m = arith.minui %d, %acc : i32
    linalg.yield %m : i32
  } -> tensor<8192xi32>

  return %best : tensor<8192xi32>
}
