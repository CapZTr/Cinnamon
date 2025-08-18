################################################################################
# PuDIncGen
#
# MLIR PuD dialect generated includes.
################################################################################

add_custom_target(PuDIncGen)

# Attributes, Dialect, Operations and Types.
add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Codegen)
