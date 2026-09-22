// Elementwise multiply. linalg named elementwise ops are type-homogeneous, so
// this is a 16x16 multiply whose high half is dropped; the FULL 2n-bit
// bits.mul_i is still charged (over-reports rather than under-reports).
func.func @vmul(%a: tensor<1048576xi16>, %b: tensor<1048576xi16>, %c: tensor<1048576xi16>) -> tensor<1048576xi16> {
  %0 = linalg.mul ins(%a, %b : tensor<1048576xi16>, tensor<1048576xi16>) outs(%c : tensor<1048576xi16>) -> tensor<1048576xi16>
  return %0 : tensor<1048576xi16>
}
