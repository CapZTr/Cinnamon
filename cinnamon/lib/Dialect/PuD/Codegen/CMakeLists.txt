add_mlir_dialect_library(PuDCodegen
  AddressAllocator.cpp
  OperandTracker.cpp
  ProgramParser.cpp

  DEPENDS
    PuDIncGen

  LINK_LIBS PUBLIC
    MLIRIR 
    MLIRFuncDialect
)
