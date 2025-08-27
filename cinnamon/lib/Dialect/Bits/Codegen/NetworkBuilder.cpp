#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>

#include <cassert>
#include <mockturtle/algorithms/cleanup.hpp>


using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : module(module) {}

LogicalResult NetworkBuilder::build() {
  SmallVector<Operation *> pendingBinaryOps;
  AssembleOp assemble;

  module.walk([&](Operation* op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      auto slice = transpose.getOutput();
      migSignalMap[slice] = mig.create_pi();
      inputSlices.push_back(slice);
    } else if (auto add = dyn_cast<AddOp>(op)) {
      pendingBinaryOps.push_back(op);
    } else if (auto assembleOp = dyn_cast<AssembleOp>(op)) {
      assemble = assembleOp;
    }
  });

  bool progress = true;
  while (progress && !pendingBinaryOps.empty()) {
    progress = false;

    for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
      if (operandsBuilt(*it)) {
        if (auto add = dyn_cast<AddOp>(*it)) {
          auto lhs = migSignalMap.lookup(add.getLhs());
          auto rhs = migSignalMap.lookup(add.getRhs());
          auto cin = mig.create_pi();
          auto [sum, cout] = buildAdd(lhs, rhs, cin);
          migSignalMap[add.getResult()] = sum;
          const int cinIdex = mig.num_pis() - 1;
          assert(!carryMap.contains(cinIdex));
          carryMap[cinIdex] = mig.num_pos() - 1;

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
    llvm::errs() << "NetworkBuilder: Some binary ops could not be parsed "
        "due to missing operands\n";
    return failure();
  }

  auto result = assemble.getInput();
  mig.create_po(migSignalMap.lookup(result));
  outputSlices.push_back(result);

  mig = mockturtle::cleanup_dangling(mig);

  return success();
}

std::pair<MIG::signal, MIG::signal> NetworkBuilder::buildAdd(
    MIG::signal const& lhs,
    MIG::signal const& rhs,
    MIG::signal const& cin) {
  auto cout = mig.create_maj(lhs, rhs, cin);
  mig.create_po(cout);
  auto maj = mig.create_maj(lhs, rhs, mig.create_not(cin));
  auto sum = mig.create_maj(maj, cin, mig.create_not(cout));
  return {sum, cout};
}

bool NetworkBuilder::operandsBuilt(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!migSignalMap.count(operand))
      return false;
  }
  return true;
}
