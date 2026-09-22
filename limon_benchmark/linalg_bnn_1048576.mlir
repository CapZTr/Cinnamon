// Binary neural network dot-product kernel: XNOR + popcount, fused.
//
// With weights and activations binarized to +/-1 and encoded in one bit
// (0 -> -1, 1 -> +1), a binary multiply IS an XNOR and the K-element dot
// product is  2 * popcount(xnor(x, w)) - K.  This kernel computes the
// popcount(xnor(...)) core; the affine  2p - K  tail is a per-output scalar
// transform that costs one add against the 1534 row ops the core costs.
//
// 2^20 dot products, each over K = 64 binary values (one i64 word), i.e.
// 2^26 binary MACs.  This is ONE K=64 reduction tile, not a whole layer.
//
// DO NOT read the row-op counts from this kernel as a layer cost.  A
// binarized 1024 -> 1024 FC layer over a batch of 1024 has the same 2^20
// outputs, but K = 1024 = 16 such tiles, so 2^30 MACs -- 16x this kernel,
// plus the 15 cross-tile accumulate steps that chain the partial counts.
// Extrapolating from this trace without that factor underestimates the
// layer by 16x.
//
// Relatedly, the i8 result type is correct ONLY for a single tile: one
// tile's count maxes at 64 (7 bits), while a 16-tile accumulation reaches
// 1024 and needs at least 11.  A multi-tile variant must widen the
// accumulator; it cannot just chain this kernel's output type.
//
// The single-tile framing is deliberate -- it is what keeps the XNOR
// isolated (see the popcount note below) -- but it measures the per-tile
// core, and the paper must multiply it out explicitly.
//
// MLIR has no xnor op, so the body uses the double-xori  (a ^ b) ^ -1  that
// every binary-network kernel in the wild is written with.  Element count is
// chosen so it tiles into 128 whole 8192-lane bank rows.
//
// Deliberately the SAME element count as benchmark/popcount/i64 so the two
// traces differ by exactly the XNOR: any delta in row ops is attributable to
// that one op and nothing else.

#map = affine_map<(d0) -> (d0)>

func.func @bnn_xnor_popcount(%act: tensor<1048576xi64>,
                             %wgt: tensor<1048576xi64>) -> tensor<1048576xi8> {
  %ones = arith.constant -1 : i64
  %out = tensor.empty() : tensor<1048576xi8>
  %r = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%act, %wgt : tensor<1048576xi64>, tensor<1048576xi64>)
    outs(%out : tensor<1048576xi8>) {
  ^bb0(%x: i64, %w: i64, %o: i8):
    %xor  = arith.xori %x, %w : i64
    %xnor = arith.xori %xor, %ones : i64
    %pc   = math.ctpop %xnor : i64
    %pct  = arith.trunci %pc : i64 to i8
    linalg.yield %pct : i8
  } -> tensor<1048576xi8>
  return %r : tensor<1048576xi8>
}
