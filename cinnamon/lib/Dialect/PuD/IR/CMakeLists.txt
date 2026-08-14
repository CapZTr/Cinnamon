add_mlir_dialect_library(PuDIR
    PuDAttributes.cpp
    PuDBase.cpp
    PuDOps.cpp
    PuDTypes.cpp

  DEPENDS
    PuDIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
)
