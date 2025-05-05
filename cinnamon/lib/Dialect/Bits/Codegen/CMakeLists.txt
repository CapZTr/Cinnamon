add_mlir_dialect_library(BitsCodegen
  NetworkBuilder.cpp

  DEPENDS
    BitsIncGen

  LINK_LIBS PUBLIC
    MLIRIR 
    MLIRFuncDialect
    mockturtle
)
