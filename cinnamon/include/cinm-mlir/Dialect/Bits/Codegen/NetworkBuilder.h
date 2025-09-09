#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mockturtle/networks/mig.hpp>

using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:
  explicit NetworkBuilder(ModuleOp module);

  LogicalResult build();

  const MIG &getNetwork() const { return mig; }

  const MIG &getMulFSignNtk() const { return mulFSignNtk; }

  bool isMulFNtk() { return isMulF; }

  const DenseMap<int, int> &getCarryMap() const { return carryMap; }

  ArrayRef<TypedValue<SliceType>> getInputSlices() const { return inputSlices; }

  ArrayRef<TypedValue<SliceType>> getOutputSlices() const {
    return outputSlices;
  }

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
  MIG mulFSignNtk;
  bool isMulF = false;
  DenseMap<Value, MIG::signal> migSignalMap;
  DenseMap<Value, MIG::signal> mulFSignSignalMap;
  DenseMap<int, int> carryMap;
  SmallVector<TypedValue<SliceType>> inputSlices;
  SmallVector<TypedValue<SliceType>, 1> outputSlices;

  std::pair<MIG::signal, MIG::signal> buildAdd(MIG::signal const &lhs,
                                               MIG::signal const &rhs,
                                               MIG::signal const &cin);

  bool operandsBuilt(Operation *op) const;
};

}
