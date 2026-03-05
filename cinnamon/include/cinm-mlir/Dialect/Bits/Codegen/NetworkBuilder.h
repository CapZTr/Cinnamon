#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mockturtle/networks/mig.hpp>

using MIG = mockturtle::mig_network;

namespace mlir::bits {

class NetworkBuilder {
public:
  struct SubgraphNetwork {
    int64_t bankId = 0;
    SmallVector<int64_t> nodeIds;
    SmallVector<int64_t> inputValueIds;
    SmallVector<int64_t> outputValueIds;
    MIG mig;
    MIG mulFSignNtk;
    MIG mulFExponentNtk;
    MIG gtNtk;
    MIG ltNtk;
    bool isMulF = false;
    bool isMax = false;
    bool isMin = false;
    DenseMap<int, int> carryMap;
  };

  struct SubgraphDependency {
    size_t producerBankId = 0;
    size_t consumerBankId = 0;
    SmallVector<int64_t> valueIds;

    void print(llvm::raw_ostream &os) const {
      os << "SubgraphDependency { producer=" << producerBankId
         << ", consumer=" << consumerBankId << ", values=[";
      for (size_t i = 0; i < valueIds.size(); ++i) {
        if (i > 0) {
          os << ", ";
        }
        os << valueIds[i];
      }
      os << "] }\n";
    }
  };

  explicit NetworkBuilder(ModuleOp module);
  explicit NetworkBuilder(func::FuncOp func);

  LogicalResult build();

  const MIG &getNetwork() const { return mig; }

  const MIG &getMulFSignNtk() const { return mulFSignNtk; }

  const MIG &getMulFExponentNtk() const { return mulFExponentNtk; }

  const MIG &getGTNtk() const { return gtNtk; }

  const MIG &getLTNtk() const { return ltNtk; }

  bool isMulFNtk() { return isMulF; }

  bool isMaxNtk() { return isMax; }

  bool isMinNtk() { return isMin; }

  const DenseMap<int, int> &getCarryMap() const { return carryMap; }

  ArrayRef<Value> getInputSlices() const { return inputSlices; }

  ArrayRef<Value> getOutputSlices() const {
    return outputSlices;
  }

  bool hasSubgraphs() const { return !subgraphNetworks.empty(); }

  ArrayRef<SubgraphNetwork> getSubgraphNetworks() const {
    return subgraphNetworks;
  }

  ArrayRef<SubgraphDependency> getSubgraphDependencies() const {
    return subgraphDependencies;
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
  func::FuncOp func;
  MIG mig;
  MIG mulFSignNtk;
  MIG mulFExponentNtk;
  MIG gtNtk;
  MIG ltNtk;
  bool isMulF = false;
  bool isMax = false;
  bool isMin = false;
  DenseMap<Value, MIG::signal> migSignalMap;
  DenseMap<Value, MIG::signal> mulFSignSignalMap;
  DenseMap<Value, MIG::signal> gtSignalMap;
  DenseMap<Value, MIG::signal> ltSignalMap;
  DenseMap<int, int> carryMap;
  SmallVector<Value> inputSlices;
  SmallVector<Value, 1> outputSlices;
  SmallVector<SubgraphNetwork, 0> subgraphNetworks;
  SmallVector<SubgraphDependency> subgraphDependencies;

  std::pair<MIG::signal, MIG::signal> buildAdd(MIG &ntk,
                                               MIG::signal const &lhs,
                                               MIG::signal const &rhs,
                                               MIG::signal const &cin);
  MIG::signal buildMux2(MIG &ntk,
                        MIG::signal const &s,
                        MIG::signal const &lhs,
                        MIG::signal const &rhs);

};

}
