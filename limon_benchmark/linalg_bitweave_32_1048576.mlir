// BitWeaving/V BETWEEN range scan over a bit-sliced database column.
//
//   SELECT COUNT(*) FROM col WHERE 1000000 < value < 2000000
//
// 2^20 values of 32 bits.  The column is stored bit-sliced (BitWeaving/V):
// bit plane j of every value lives in row j, so one predicate evaluation is
// 8192-way SIMD across a bank row by construction.  bits.transpose of the
// column tensor produces exactly that layout, which is why the whole scan
// lowers to a single bits.range_scan.
//
// Cost is 4*32 + 5 row ops per 8192-value tile -- strictly LINEAR in the
// column width, which is BitWeaving's central claim over a fixed-width scan.
// The i8/i16/i32 shapes exist to show that line.
//
// The COUNT(*) is left to the host: counting a mask across lanes is a
// horizontal reduction, awkward on PuD and done on the CPU by all three
// reference implementations in MIMDRAM/microworkloads.
//
// No early termination -- real BitWeaving/V may stop once every value is
// resolved, but that is data dependent and unrepresentable in a static trace.
// All three references also scan every plane unconditionally, so this matches
// them and is an upper bound on the scan cost.

#map = affine_map<(d0) -> (d0)>

func.func @bitweave_between(%col: tensor<1048576xi32>) -> tensor<1048576xi1> {
  %lo = arith.constant 1000000 : i32
  %hi = arith.constant 2000000 : i32
  %out = tensor.empty() : tensor<1048576xi1>
  %m = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%col : tensor<1048576xi32>) outs(%out : tensor<1048576xi1>) {
  ^bb0(%v: i32, %o: i1):
    %a = arith.cmpi sgt, %v, %lo : i32
    %b = arith.cmpi slt, %v, %hi : i32
    %p = arith.andi %a, %b : i1
    linalg.yield %p : i1
  } -> tensor<1048576xi1>
  return %m : tensor<1048576xi1>
}
