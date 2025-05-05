#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>

#include <mockturtle/algorithms/cleanup.hpp>

using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : module(module) {}

LogicalResult NetworkBuilder::build() {
  SmallVector<Operation *> pendingBinaryOps;
  SmallVector<AssembleOp> assembles;

  module.walk([&](Operation *op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      migSignalMap[transpose.getOutput()] = mig.create_pi();
    } else if (auto add = dyn_cast<AddOp>(op)) {
      if (operandsBuilt(op)) {
        buildAdd(add);
      } else {
        pendingBinaryOps.push_back(op);
      }
    } else if (auto assemble = dyn_cast<AssembleOp>(op)) {
      assembles.push_back(assemble);
    }
  });

  bool progress = true;
  while (progress && !pendingBinaryOps.empty()) {
    progress = false;

    for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
      if (operandsBuilt(*it)) {
        if (auto add = dyn_cast<AddOp>(*it)) {
          buildAdd(add);
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else {
          emitError((*it)->getLoc(),
                    "NetworkBuilder: Unsupported type of binary op");
          return failure();
        }
      } else {
        ++it;
      }
    }
  }

  if (!pendingBinaryOps.empty()) {
    llvm::errs() << "NetworkBuilder: Some binary ops could not be parsed due to missing operands\n";
    return failure();
  }

  for (auto assemble : assembles) {
    mig.create_po(migSignalMap.lookup(assemble.getInput()));
  }

  mig = mockturtle::cleanup_dangling(mig);

  debugPrint();

  return success();
}

void NetworkBuilder::buildAdd(AddOp add) {
  auto lhs = migSignalMap.lookup(add.getLhs());
  auto rhs = migSignalMap.lookup(add.getRhs());
  auto cin = mig.get_constant(false);

  auto cout = mig.create_maj(lhs, rhs, cin);
  auto maj = mig.create_maj(lhs, rhs, mig.create_not(cin));
  auto sum = mig.create_maj(maj, cin, mig.create_not(cout));

  migSignalMap[add.getResult()] = sum;
}

bool NetworkBuilder::operandsBuilt(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!migSignalMap.count(operand))
      return false;
  }
  return true;
}

void NetworkBuilder::debugPrint(llvm::raw_ostream &os) {
  os << "\n=== MIG Network Debug Info ===\n";
  os << "Primary Inputs (PIs): " << mig.num_pis() << "\n";
  os << "Primary Outputs (POs): " << mig.num_pos() << "\n";
  os << "Gates: " << mig.num_gates() << "\n";
  os << "Total Nodes: " << mig.size() << "\n\n";

  os << "=== Input Signal Map ===\n";
  mig.foreach_pi([&](auto input) {
    os << "  Input: Node " << input << "\n";
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
