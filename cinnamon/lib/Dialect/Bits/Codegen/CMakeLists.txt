add_mlir_dialect_library(BitsCodegen
  BitsParser.cpp
  NetworkBuilder.cpp

  LINK_LIBS PUBLIC
      MLIRIR 
      mockturtle
)
