add_mlir_dialect_library(BitsTransforms
  SliceToBitPass.cpp
  LimeOptimizationPass.cpp

  DEPENDS
    BitsIncGen
    BitsTransformPassesIncGen
  
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRSCFDialect
)

target_include_directories(BitsTransforms PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/include
  ${CMAKE_SOURCE_DIR}/extern/lime/src
)

file(GLOB LIB_DIRS "${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/lib/*")

foreach(dir ${LIB_DIRS})
  if(IS_DIRECTORY "${dir}")
    target_include_directories(BitsTransforms PRIVATE ${dir})
  endif()
endforeach()

target_link_libraries(BitsTransforms PRIVATE lime_base)
