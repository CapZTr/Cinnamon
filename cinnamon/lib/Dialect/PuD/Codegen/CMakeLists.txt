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

# mockturtle now lives under lime/eggmock/mockturtle and exports its headers
# (plus kitty/lorina/fmt/... ) through the `eggmock` interface target, so linking
# lime_base is enough; only lime's own headers (gp.h, utils.h) need a raw path.
target_include_directories(PuDCodegen PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/src
)

target_link_libraries(PuDCodegen PUBLIC cinm::fmt lime_base)
target_compile_definitions(PuDCodegen PRIVATE FMT_HEADER_ONLY)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/InstructionMapper.cpp
  PROPERTIES COMPILE_FLAGS "-frtti"
)
