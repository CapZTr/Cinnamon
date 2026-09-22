func.func @gemv(
    %matrix: tensor<2048x2048xi8>, %vector: tensor<2048xi8>, %result: tensor<2048xi16>)
    -> tensor<2048xi16> {
  %res = linalg.matvec
    ins(%matrix, %vector : tensor<2048x2048xi8>, tensor<2048xi8>)
    outs(%result : tensor<2048xi16>) -> tensor<2048xi16>
  return %res : tensor<2048xi16>
}
