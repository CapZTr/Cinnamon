func.func @relu(%a: tensor<1048576xi16>) -> tensor<1048576xi16> {
  %c0 = arith.constant 0 : i16
  %e = tensor.empty() : tensor<1048576xi16>
  %zero = linalg.fill ins(%c0 : i16) outs(%e : tensor<1048576xi16>) -> tensor<1048576xi16>
  %o = tensor.empty() : tensor<1048576xi16>
  %r = linalg.max ins(%a, %zero : tensor<1048576xi16>, tensor<1048576xi16>) outs(%o : tensor<1048576xi16>) -> tensor<1048576xi16>
  return %r : tensor<1048576xi16>
}
