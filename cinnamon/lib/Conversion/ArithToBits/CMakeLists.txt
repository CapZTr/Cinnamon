add_mlir_conversion_library(MLIRArithToBits
  ArithToBits.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/cim/Conversion/ArithToBits

  DEPENDS
    BitsFrontendConversionPassIncGen
  
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
)
