// Elementwise subtract, the same shape as the add: the
// smallest thing that exercises bits.sub_i end to end.
func.func @vsub(%a: tensor<1048576xi16>, %b: tensor<1048576xi16>, %c: tensor<1048576xi16>) -> tensor<1048576xi16> {
  %0 = linalg.sub ins(%a, %b : tensor<1048576xi16>, tensor<1048576xi16>) outs(%c : tensor<1048576xi16>) -> tensor<1048576xi16>
  return %0 : tensor<1048576xi16>
}
