set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BitsTransforms)
add_public_tablegen_target(BitsTransformPassesIncGen)

add_mlir_doc(Passes BitsTransformPasses ./ -gen-pass-doc)
