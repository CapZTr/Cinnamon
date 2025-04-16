module {
  func.func @main(%arg0: tensor<64xi32>, %arg1: tensor<64xi32>, %arg2: tensor<64xi32>, %arg3: tensor<64xi32>) -> !bits.slice<32x64> {
    %input0 = bits.transpose %arg0 : tensor<64xi32> -> !bits.slice<32x64>
    %input1 = bits.transpose %arg1 : tensor<64xi32> -> !bits.slice<32x64>
    %sum1 = bits.add %input0, %input1 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>
    // %input2 = bits.transpose %arg2 : tensor<64xi32> -> !bits.slice<32x64>
    // %sum2 = bits.add %sum1, %input2 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>
    // %input3 = bits.transpose %arg3 : tensor<64xi32> -> !bits.slice<32x64>
    // %result = bits.sub %sum2, %input3 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>

    return %sum1 : !bits.slice<32x64>
    // return %result : !bits.slice<32x64>
  }
}
