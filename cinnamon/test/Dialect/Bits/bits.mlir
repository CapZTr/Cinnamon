module {
  func.func @add_slices() -> !bits.slice<32x64> {
    %bw = arith.constant 32 : i64
    %vl = arith.constant 64 : i64
    %bi = arith.constant 0 : i64
    %vi = arith.constant 1 : i64
    %a = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15,
                               16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 26, 27, 28, 29, 30, 31,
                               32, 33, 34, 35, 36, 37, 38, 39,
                               40, 41, 42, 43, 44, 45, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55,
                               56, 57, 58, 59, 60, 61, 62, 63]>
         : tensor<64xi32>
    
    %b = arith.constant dense<[1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1]>
         : tensor<64xi32>

    %sa = bits.transpose %a : tensor<64xi32> -> !bits.slice<32x64>
    %sb = bits.transpose %b : tensor<64xi32> -> !bits.slice<32x64>
    %sc = bits.create_slice %bw, %vl : i64, i64 -> !bits.slice<32x64>
    %ba = bits.extract %sa[%bi, %vi] : !bits.slice<32x64>, i64, i64 -> !bits.bit
    %bb = bits.extract %sb[%bi, %vi] : !bits.slice<32x64>, i64, i64 -> !bits.bit
    %bc = bits.create_bit : !bits.bit

    %sum_bit, %cout = bits.add_bit %ba, %bb, %bc : !bits.bit, !bits.bit, !bits.bit -> !bits.bit, !bits.bit

    %sd = bits.insert %sum_bit -> %sc[%bi, %vi] : !bits.bit, !bits.slice<32x64>, i64, i64 -> !bits.slice<32x64>

    %sum1 = bits.add %sa, %sb : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>

    %sum2 = bits.sub %sa, %sb : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>

    %sum3 = bits.add %sum1, %sum2 : !bits.slice<32x64>, !bits.slice<32x64> -> !bits.slice<32x64>

    return %sum3 : !bits.slice<32x64>
  }
}
