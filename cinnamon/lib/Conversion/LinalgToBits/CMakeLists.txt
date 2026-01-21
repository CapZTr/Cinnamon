add_mlir_conversion_library(MLIRLinalgToBits
  LinalgToBits.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/bits_frontend/Conversion/LinalgToBits

  DEPENDS
    BitsFrontendConversionPassIncGen
    BitsIncGen
  
  LINK_COMPONENTS
    Core
  
  LINK_LIBS PUBLIC
    BitsIR
    MLIRDialectUtils
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
