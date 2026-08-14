add_mlir_conversion_library(MLIRBitsToPuD
  BitsToPuD.cpp
  BitsRowsToPuD.cpp
  ${CMAKE_SOURCE_DIR}/lib/Dialect/Bits/Codegen/NetworkBuilder.cpp
  ${CMAKE_SOURCE_DIR}/lib/Dialect/PuD/Codegen/AddressAllocator.cpp
  ${CMAKE_SOURCE_DIR}/lib/Dialect/PuD/Codegen/OperandTracker.cpp
  ${CMAKE_SOURCE_DIR}/lib/Dialect/PuD/Codegen/ProgramParser.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/bits/Conversion/BitsToPuD

  DEPENDS
    BitsConversionPassIncGen
    PuDIncGen
  
  LINK_COMPONENTS
    Core
  
  LINK_LIBS PUBLIC
    BitsIR
    MLIRDialectUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
    MLIRSCFDialect
    PuDIR
    BitsCodegen
)

target_link_libraries(MLIRBitsToPuD PUBLIC PuDCodegen)

# mockturtle now lives under lime/eggmock/mockturtle and exports its headers
# (plus kitty/lorina/fmt/... ) through the `eggmock` interface target, so linking
# lime_base is enough; only lime's own headers (gp.h, utils.h) need a raw path.
target_include_directories(MLIRBitsToPuD PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/src
)

target_link_libraries(MLIRBitsToPuD PUBLIC cinm::fmt)
# target_link_libraries(MLIRBitsToPuD PRIVATE lime_base fmt-local)
target_link_libraries(MLIRBitsToPuD PRIVATE lime_base)
