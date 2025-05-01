#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include <mlir/IR/Value.h>
#include <mockturtle/networks/aig.hpp>
#include <mockturtle/networks/mig.hpp>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <optional>
#include <utility>

using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:
  explicit NetworkBuilder(ModuleOp module);

  std::optional<MIG> build();

private:
  BitsParser parser;
  MIG mig;
  llvm::DenseMap<Value, MIG::signal> migSignalMap;

  void buildAdd(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result);
  std::pair<MIG::signal, MIG::signal> createFullAdderInMIG(
      const MIG::signal a, const MIG::signal b, const MIG::signal cin);
  void debugPrintMIGSignals(llvm::raw_ostream &os = llvm::errs());
};

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H

