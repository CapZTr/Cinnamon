################################################################################
# BitsIncGen
#
# MLIR Bits dialect generated includes.
################################################################################

add_custom_target(BitsIncGen)

# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Codegen)
