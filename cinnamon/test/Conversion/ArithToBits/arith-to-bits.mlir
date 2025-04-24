module {
  func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<64xi32>) -> tensor<64xi32> {
    %sum = arith.addi %arg0, %arg1 : tensor<64xi32>
    return %sum : tensor<64xi32>
  }
}
