// Control group for the popcount benefit kernel: plain elementwise add over
// the same element count. Tiled into 1048576/8192 = 128 full-row waves.
func.func @vadd(%a: tensor<1048576xi16>, %b: tensor<1048576xi16>, %c: tensor<1048576xi16>) -> tensor<1048576xi16> {
  %0 = linalg.add ins(%a, %b : tensor<1048576xi16>, tensor<1048576xi16>) outs(%c : tensor<1048576xi16>) -> tensor<1048576xi16>
  return %0 : tensor<1048576xi16>
}
