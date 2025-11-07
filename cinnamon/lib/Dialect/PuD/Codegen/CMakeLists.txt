add_mlir_dialect_library(PuDCodegen
  AddressAllocator.cpp
  InstructionMapper.cpp
  OperandTracker.cpp
  ProgramParser.cpp

  DEPENDS
    PuDIncGen

  LINK_LIBS PUBLIC
    MLIRIR 
    MLIRFuncDialect
)

target_include_directories(PuDCodegen PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/include
  ${CMAKE_SOURCE_DIR}/extern/lime/src
  ${CMAKE_SOURCE_DIR}/extern/lime/eggmock/include
)

file(GLOB LIB_DIRS "${CMAKE_SOURCE_DIR}/extern/lime/mockturtle/lib/*")

foreach(dir ${LIB_DIRS})
  if(IS_DIRECTORY "${dir}")
    target_include_directories(PuDCodegen PRIVATE ${dir})
  endif()
endforeach()

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/InstructionMapper.cpp
  PROPERTIES COMPILE_FLAGS "-frtti"
)
