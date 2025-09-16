#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
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
      auto elemType = cast<RankedTensorType>(transpose.getInput().getType())
          .getElementType();
      migSignalMap[slice] = mig.create_pi();
      if (isa<FloatType>(elemType)) {
        mulFSignSignalMap[slice] = mulFSignNtk.create_pi();
      }
      inputSlices.push_back(slice);
    } else if (auto add = dyn_cast<AddIOp>(op)) {
      pendingBinaryOps.push_back(op);
    } else if (auto mul = dyn_cast<MulFOp>(op)) {
      pendingBinaryOps.push_back(op);
    }
    else if (auto assembleOp = dyn_cast<AssembleOp>(op)) {
      assemble = assembleOp;
    }
  });

  bool allAddI = llvm::all_of(pendingBinaryOps, [](Operation *op) {
    return isa<AddIOp>(op);
  });
  bool allMulF = llvm::all_of(pendingBinaryOps, [](Operation *op) {
    return isa<MulFOp>(op);
  });
  if (allMulF) {
    isMulF = true;
  }

  assert((allAddI || allMulF) &&
      "pendingBinaryOps must contain either all AddIOp or all MulFOp");

  bool progress = true;
  while (progress && !pendingBinaryOps.empty()) {
    progress = false;

    for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
      if (operandsBuilt(*it)) {
        if (auto add = dyn_cast<AddIOp>(*it)) {
          auto lhs = migSignalMap.lookup(add.getLhs());
          auto rhs = migSignalMap.lookup(add.getRhs());
          auto cin = mig.create_pi();
          auto [sum, cout] = buildAdd(mig, lhs, rhs, cin);
          migSignalMap[add.getResult()] = sum;
          const int cinIdex = mig.num_pis() - 1;
          assert(!carryMap.contains(cinIdex));
          carryMap[cinIdex] = mig.num_pos() - 1;

          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto mul = dyn_cast<MulFOp>(*it)) {
          // sign
          const auto &lhsSign = mulFSignSignalMap[mul.getLhs()];
          const auto &rhsSign = mulFSignSignalMap[mul.getRhs()];
          mulFSignSignalMap[mul.getResult()] = mulFSignNtk.create_xor(
              lhsSign, rhsSign);

          // exponent and mantissa
          auto lhs = migSignalMap.lookup(mul.getLhs());
          auto rhs = migSignalMap.lookup(mul.getRhs());
          auto cin = mig.create_pi();
          auto [sum, cout] = buildAdd(mig, lhs, rhs, cin);
          migSignalMap[mul.getResult()] = sum;
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
  mig.create_po(migSignalMap[result]);
  if (isMulF) {
    assert(mulFSignSignalMap.contains(result));
    mulFSignNtk.create_po(mulFSignSignalMap[result]);

    // Exponent Network
    auto i0 = mulFExponentNtk.create_pi();
    auto i1 = mulFExponentNtk.create_pi();
    auto i2 = mulFExponentNtk.create_pi();
    auto [sum, cout] = buildAdd(mulFExponentNtk, i0, i1, i2);
    mulFExponentNtk.create_po(sum);
  }
  outputSlices.push_back(result);

  mig = mockturtle::cleanup_dangling(mig);
  if (isMulF) {
    mulFSignNtk = mockturtle::cleanup_dangling(mulFSignNtk);
    mulFExponentNtk = mockturtle::cleanup_dangling(mulFExponentNtk);
  }

  return success();
}

std::pair<MIG::signal, MIG::signal> NetworkBuilder::buildAdd(
    MIG &ntk,
    MIG::signal const &lhs,
    MIG::signal const &rhs,
    MIG::signal const &cin) {
  auto cout = ntk.create_maj(lhs, rhs, cin);
  ntk.create_po(cout);
  auto maj = ntk.create_maj(lhs, rhs, ntk.create_not(cin));
  auto sum = ntk.create_maj(maj, cin, ntk.create_not(cout));
  return {sum, cout};
}

bool NetworkBuilder::operandsBuilt(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!migSignalMap.count(operand))
      return false;
  }
  return true;
}
