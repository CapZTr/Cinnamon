#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include "mockturtle/networks/aig.hpp"
#include "mockturtle/networks/mig.hpp"

#include <llvm/ADT/DenseMap.h>
#include <string>
#include <utility>
#include <vector>

using AIG = mockturtle::aig_network;
using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:

  explicit NetworkBuilder(ModuleOp module);

  void build(const std::string &filePathAig, const std::string &filePathMig);

private:
  BitsParser parser;
  AIG aig;
  MIG mig;
  llvm::DenseMap<BitplaneData, std::vector<AIG::signal>> aigSignalMap;
  llvm::DenseMap<BitplaneData, std::vector<MIG::signal>> migSignalMap;

  void buildInputs(const BitplaneData &data);
  void buildAdd(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result);
  void buildSub(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result);
  void buildOutputs(const BitplaneData &output);
  std::pair<MIG::signal, MIG::signal> createFullAdderInMIG(const MIG::signal a, const MIG::signal b, const MIG::signal cin);
};

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

