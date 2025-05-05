#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H


#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>

#include <mockturtle/networks/mig.hpp>

using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:
  explicit NetworkBuilder(ModuleOp module);

  LogicalResult build();

  const MIG &getNetwork() const { return mig; }

private:
  ModuleOp module;
  MIG mig;
  DenseMap<Value, MIG::signal> migSignalMap;

  void buildAdd(AddOp add);

  bool operandsBuilt(Operation *op) const;

  void debugPrint(llvm::raw_ostream &os = llvm::errs());
};

}

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
