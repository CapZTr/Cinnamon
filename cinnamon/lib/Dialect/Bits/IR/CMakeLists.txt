add_mlir_dialect_library(BitsIR
        BitsBase.cpp
        BitsOps.cpp
        BitsTypes.cpp

    DEPENDS
        BitsIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
