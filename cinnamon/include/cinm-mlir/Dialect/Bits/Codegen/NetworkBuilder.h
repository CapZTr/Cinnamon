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

  const MIG& getNetwork() const { return mig; }

  const SmallVector<Value> &getInputSlices() const { return inputSlices; }

  static void debugPrint(MIG const &mig, llvm::raw_ostream &os = llvm::errs()) {
    os << "\n=== MIG Network Debug Info ===\n";
    os << "Primary Inputs (PIs): " << mig.num_pis() << "\n";
    os << "Primary Outputs (POs): " << mig.num_pos() << "\n";
    os << "Gates: " << mig.num_gates() << "\n";
    os << "Total Nodes: " << mig.size() << "\n\n";

    os << "=== Input Signal Map ===\n";
    mig.foreach_pi([&](auto input) {
      os << "  Input: Node " << input << " : " << mig.pi_index(input) << "\n";
    });

    os << "\n=== Node Details ===\n";
    mig.foreach_node([&](auto node) {
      if (mig.is_pi(node)) return;

      os << "  Node " << node << ": ";
      
      if (mig.is_maj(node)) {
        os << "MAJ( ";
        mig.foreach_fanin(node, [&](auto const& fanin, auto i) {
          if (i > 0) os << ", ";
          os << mig.get_node(fanin) 
            << (mig.is_complemented(fanin) ? "'" : "");
        });
        os << " )";
      } else if (mig.is_constant(node)) {
        os << "Const";
      }
      os << "\n";
    });

    os << "\n=== Output Signal Map ===\n";
    mig.foreach_po([&](auto output) {
      os << "  Output: Node " << mig.get_node(output) << "\n";
    });

    os << "===========================\n\n";
  }

private:
  ModuleOp module;
  MIG mig;
  DenseMap<Value, MIG::signal> migSignalMap;
  DenseMap<AddOp, MIG::signal> coutMap;
  SmallVector<Value> inputSlices;

  std::optional<MIG::signal> findCarryIn(AddOp add);
  std::pair<MIG::signal, MIG::signal> buildAdd(MIG::signal const& lhs,
                                               MIG::signal const& rhs,
                                               MIG::signal const& cin);

  bool operandsBuilt(Operation *op) const;
};

}

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_NETWORKBUILDER_H
