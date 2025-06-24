add_mlir_conversion_library(MLIRBitsToPuD
  BitsToPuD.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/bits/Conversion/BitsToPuD

  DEPENDS
    BitsConversionPassIncGen
    PuDIncGen
  
  LINK_COMPONENTS
    Core
  
  LINK_LIBS PUBLIC
    BitsIR
    MLIRDialectUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
    PuDIR
)
