#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/mig.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <string>

using AIG = mockturtle::aig_network;
using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:
  explicit NetworkBuilder(ModuleOp module);

  LogicalResult build(const std::string &filePathAig, const std::string &filePathMig);

private:
  BitsParser parser;
  AIG aig;
  MIG mig;
  llvm::DenseMap<BitplaneData, AIG::signal> aigSignalMap;
  llvm::DenseMap<BitplaneData, MIG::signal> migSignalMap;

  void buildAdd(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result);
  void buildSub(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result);
};

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

