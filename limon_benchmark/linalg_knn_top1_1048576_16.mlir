// KNN top-1 selection over a binary (Hamming) similarity matrix.
//
// This is the SELECTION half of a k-nearest-neighbour search. The similarity
// half is the BNN kernel: with LSH- or PQ-derived binary codes, similarity is
// popcount(xnor(query, dbvec)), so "nearest" is "largest xnor-popcount" and
// benchmark/bnn/i64 already computes it. Distance therefore costs nothing new
// here; what is new is the top-k selection.
//
// LAYOUT, which is the whole design: the LANE dimension carries the queries
// and the reduction walks the database, so every lane maintains its own
// running best independently -- a pure per-lane operation. Putting the
// database in the lanes instead would make top-k a cross-lane horizontal
// reduction, the one shape a PuD substrate is bad at, and is why the popcount
// and BitWeaving counts are left to the host.
//
// 2^20 queries x 16 database vectors, k=1.
//
// DO NOT read this as a whole search. 16 database vectors is ONE SHARD: a
// 2^20-vector database is 65536 such invocations, plus a merge of their
// per-shard bests. Extrapolating a full search from this trace without that
// factor underestimates it by the shard count.
//
// Why binary and not float/int L2: with an i8 L2 distance the top-k would be
// ~0.5% of the kernel and the other 99.5% would be a GEMM we already measure.
// Binary similarity makes selection ~a third of the work and reuses BNN whole.

#m0 = affine_map<(d0, d1) -> (d0, d1)>
#m1 = affine_map<(d0, d1) -> (d0)>

func.func @knn_top1(%sim: tensor<1048576x16xi8>) -> tensor<1048576xi8> {
  %init = tensor.empty() : tensor<1048576xi8>
  %best = linalg.generic {
    indexing_maps = [#m0, #m1],
    iterator_types = ["parallel", "reduction"]
  } ins(%sim : tensor<1048576x16xi8>) outs(%init : tensor<1048576xi8>) {
  ^bb0(%s: i8, %acc: i8):
    %m = arith.maxui %s, %acc : i8
    linalg.yield %m : i8
  } -> tensor<1048576xi8>
  return %best : tensor<1048576xi8>
}
