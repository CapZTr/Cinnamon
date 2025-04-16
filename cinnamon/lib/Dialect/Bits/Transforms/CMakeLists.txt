add_mlir_dialect_library(BitsTransforms
  LowerArithPass.cpp

  DEPENDS
    BitsIncGen
    BitsTransformPassesIncGen
  
  LINK_LIBS PUBLIC
    MLIRIR
    MLIRArithDialect
    MLIRSCFDialect
)
