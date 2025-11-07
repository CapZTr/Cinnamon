add_mlir_conversion_library(MLIRBitsToPuD
  BitsToPuD.cpp
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
    PuDIR
)

target_link_libraries(MLIRBitsToPuD PUBLIC PuDCodegen)

target_include_directories(MLIRBitsToPuD PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/include
  ${CMAKE_SOURCE_DIR}/extern/lime/src
)

file(GLOB LIB_DIRS "${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/lib/*")

foreach(dir ${LIB_DIRS})
  if(IS_DIRECTORY "${dir}")
    target_include_directories(MLIRBitsToPuD PRIVATE ${dir})
  endif()
endforeach()

target_link_libraries(MLIRBitsToPuD PRIVATE lime_base fmt-local)
