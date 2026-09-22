// conv1x1 (shape B) followed by ReLU, fused into one trace.
func.func @conv_relu(%in: tensor<1x256x32x32xi8>, %w: tensor<64x256x1x1xi8>,
                     %out: tensor<1x64x32x32xi16>) -> tensor<1x64x32x32xi16> {
  %conv = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
    ins(%in, %w : tensor<1x256x32x32xi8>, tensor<64x256x1x1xi8>)
    outs(%out : tensor<1x64x32x32xi16>) -> tensor<1x64x32x32xi16>
  %c0 = arith.constant 0 : i16
  %e = tensor.empty() : tensor<1x64x32x32xi16>
  %zero = linalg.fill ins(%c0 : i16) outs(%e : tensor<1x64x32x32xi16>) -> tensor<1x64x32x32xi16>
  %o = tensor.empty() : tensor<1x64x32x32xi16>
  %relu = linalg.max ins(%conv, %zero : tensor<1x64x32x32xi16>, tensor<1x64x32x32xi16>)
                     outs(%o : tensor<1x64x32x32xi16>) -> tensor<1x64x32x32xi16>
  return %relu : tensor<1x64x32x32xi16>
}
