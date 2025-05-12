#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <cassert>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>

#include <mockturtle/algorithms/cleanup.hpp>
#include <optional>

using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : module(module) {}

LogicalResult NetworkBuilder::build() {
  SmallVector<Operation *> pendingBinaryOps;
  AssembleOp assemble;

  auto cin = mig.create_pi();
  bool cinUsed = false;

  module.walk([&](Operation *op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      migSignalMap[transpose.getOutput()] = mig.create_pi();
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
          MIG::signal carryIn;
          
          auto found = findCarryIn(add);
          if (found) {
            carryIn = *found;
          } else {
            if (!cinUsed) {
              carryIn = cin;
              cinUsed = true;
            } else {
              carryIn = mig.get_constant(false);
            }
          }

          auto [sum, cout] = buildAdd(lhs, rhs, carryIn);
          migSignalMap[add.getResult()] = sum;
          coutMap[add] = cout;

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

  auto result = assemble.getInput();
  mig.create_po(migSignalMap.lookup(result));

  auto finalAdd = dyn_cast<AddOp>(result.getDefiningOp());
  assert(coutMap.count(finalAdd) == 1);
  mig.create_po(coutMap.lookup(finalAdd));

  mig = mockturtle::cleanup_dangling(mig);

  return success();
}

std::optional<MIG::signal> NetworkBuilder::findCarryIn(AddOp add) {
  auto lhs = add.getLhs();
  auto rhs = add.getRhs();

  auto lhsDef = lhs.getDefiningOp();
  if (auto lhsDefAdd = dyn_cast<AddOp>(lhsDef)) {
    if (coutMap.count(lhsDefAdd)) {
      auto cin = coutMap.lookup(lhsDefAdd);
      coutMap.erase(lhsDefAdd);
      return cin;
    }
  }

  auto rhsDef = rhs.getDefiningOp();
  if (auto rhsDefAdd = dyn_cast<AddOp>(rhsDef)) {
    if (coutMap.count(rhsDefAdd)) {
      auto cin = coutMap.lookup(rhsDefAdd);
      coutMap.erase(rhsDefAdd);
      return cin;
    }
  }

  return std::nullopt;
}

std::pair<MIG::signal, MIG::signal> NetworkBuilder::buildAdd(MIG::signal const& lhs,
                                                             MIG::signal const& rhs,
                                                             MIG::signal const& cin) {
  auto cout = mig.create_maj(lhs, rhs, cin);
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
