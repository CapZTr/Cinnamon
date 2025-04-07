module {
  func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>) {
    %input0 = bits.transpose %arg0 : tensor<64xi32> -> !bits.slice<32x64>
    %input1 = bits.transpose %arg1 : tensor<64xi32> -> !bits.slice<32x64>
    %sum1 = bits.add %input0, %input1 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>
    %input2 = bits.transpose %arg2 : tensor<64xi32> -> !bits.slice<32x64>
    %sum2 = bits.add %sum1, %input2 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>
    return
  }
}
