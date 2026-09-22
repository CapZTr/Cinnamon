#map = affine_map<(d0) -> (d0)>
func.func @popcount(%a: tensor<1048576xi64>) -> tensor<1048576xi8> {
  %e = tensor.empty() : tensor<1048576xi8>
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<1048576xi64>) outs(%e : tensor<1048576xi8>) {
  ^bb0(%in: i64, %out: i8):
    %p = math.ctpop %in : i64
    %t = arith.trunci %p : i64 to i8
    linalg.yield %t : i8
  } -> tensor<1048576xi8>
  return %r : tensor<1048576xi8>
}
