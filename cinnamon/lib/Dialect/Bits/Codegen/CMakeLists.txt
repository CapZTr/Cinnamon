add_mlir_dialect_library(BitsCodegen
  NetworkBuilder.cpp
  RowOpLibrary.cpp

  DEPENDS
    BitsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    mockturtle
    lime_base
)

# only lime's own headers need a path; mockturtle and its bundled libraries come
# in through the eggmock interface target that lime_base links
target_include_directories(BitsCodegen PRIVATE
  ${CMAKE_SOURCE_DIR}/extern/lime/src
)

# lime's FFI is reached through mockturtle's network views, which need RTTI
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/RowOpLibrary.cpp
  PROPERTIES COMPILE_FLAGS "-frtti"
)
