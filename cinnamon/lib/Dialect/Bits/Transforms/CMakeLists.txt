add_mlir_dialect_library(BitsTransforms
  CubeToSlice.cpp
  OptimiseMapping.cpp

DEPENDS
  BitsIncGen
  BitsTransformPassesIncGen

LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformUtils
)

# set(GUROBI_ROOT "/home/tianruiz/gurobi/linux64")
set(GUROBI_ROOT "/opt/gurobi1300/linux64")

target_include_directories(BitsTransforms PRIVATE
  "${GUROBI_ROOT}/include"
)

target_link_libraries(BitsTransforms PRIVATE
  "${GUROBI_ROOT}/lib/libgurobi_c++.a"
  "${GUROBI_ROOT}/lib/libgurobi130.so"
)
