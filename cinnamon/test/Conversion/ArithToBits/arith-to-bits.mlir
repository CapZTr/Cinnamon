module {
  func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>, %arg3: tensor<64xi32>) -> tensor<64xi32> {
    %sum0 = arith.addi %arg0, %arg1 : tensor<64xi32>
    %sum1 = arith.addi %sum0, %arg2 : tensor<64xi32>
    %sum2 = arith.addi %sum0, %arg3 : tensor<64xi32>
    %sum3 = arith.addi %sum1, %sum2 : tensor<64xi32>
    return %sum3 : tensor<64xi32>
  }
}
