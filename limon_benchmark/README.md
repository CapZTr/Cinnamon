# limon_benchmark

The fourteen kernels the PuD compilation path is built and measured on. Plain
`linalg` on ranked tensors, no annotations — the mapping and everything below it
is the compiler's business.

Each one is compiled with
`MIMDRAM/microworkloads/pud/compile_pud_binary.sh <file> -n <banks> -m ddr|hbm`,
and all fourteen build at 8, 16, 32, 64 and 128 banks on both memories. None of
them needs a frontend option.

| name | file | symbol | what it is |
|---|---|---|---|
| `relu` | `linalg_relu_1048576.mlir` | `relu` | elementwise ReLU, 2^20 x i16. On a bit-serial substrate this is an AND against the sign bit. |
| `vadd` | `linalg_vadd_1048576.mlir` | `vadd` | elementwise add, 2^20 x i16. The simplest kernel here and the one the walkthrough uses. |
| `vsub` | `linalg_vsub_1048576.mlir` | `vsub` | elementwise subtract, same shape. Exercises the subtractor, which is the adder with one input read complemented. |
| `vmul` | `linalg_vmul_1048576.mlir` | `vmul` | elementwise multiply, same shape. Expands to shift-and-add, so it is by far the most row operations per element. |
| `popcount` | `linalg_popcount_1048576.mlir` | `popcount` | population count over 2^20 x i64, as an adder tree. |
| `bnn_i64` | `linalg_bnn_1048576.mlir` | `bnn_xnor_popcount` | binarised neural network layer, fused XNOR and popcount, packed into i64 words. |
| `bnn_i1` | `linalg_bnn_dot_8192_4096.mlir` | `bnn_dot_8192_4096` | the same idea unpacked: a binary 4096 -> 8192 fully-connected layer at i1, with a width-growing reduction tree. |
| `gemv` | `linalg_gemv_2048_2048.mlir` | `gemv` | 2048 x 2048 int8 matrix-vector product. The reduction the mapping has the most freedom with. |
| `conv1x1_relu` | `linalg_conv1x1_relu_64_256_32.mlir` | `conv_relu` | 1x1 convolution plus ReLU, 64 output channels from 256 over 32x32, int8. Two nests, the second inheriting the first's tiling. |
| `knn_top1` | `linalg_knn_top1_1048576_16.mlir` | `knn_top1` | nearest neighbour by Hamming distance, top-1. Exercises compare-then-select. |
| `knn_l2` | `linalg_knn_l2_top1_8192_256_16.mlir` | `knn_l2_top1` | nearest neighbour by squared L2, 8192 queries, int8. The largest kernel here; two nests with a reduction across them. |
| `bitweave_i8` | `linalg_bitweave_8_1048576.mlir` | `bitweave_between` | BitWeaving range scan, `lo <= x <= hi`, at i8. |
| `bitweave_i16` | `linalg_bitweave_16_1048576.mlir` | `bitweave_between` | the same at i16. |
| `bitweave_i32` | `linalg_bitweave_32_1048576.mlir` | `bitweave_between` | the same at i32. The only kernel here that is not machine learning, and the only one with no reduction at all — which makes it the control when a change is supposed to be about how reductions are mapped. |

The three `bitweave` kernels share a symbol name, so build them to output paths
of your own (`-o`) if you want all three at once.
