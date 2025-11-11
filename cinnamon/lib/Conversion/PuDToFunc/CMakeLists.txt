add_mlir_conversion_library(MLIRPuDToFunc
  PuDToFunc.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/pud/Conversion/PuDToFunc

  DEPENDS
    PuDConversionPassIncGen
  
  LINK_COMPONENTS
    Core
  
  LINK_LIBS PUBLIC
    PuDIR
    MLIRDialectUtils
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)