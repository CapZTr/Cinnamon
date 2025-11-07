#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "mockturtle/generators/arithmetic.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
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
    } else if (isa<AddIOp>(op) || isa<MulFOp>(op) || isa<AndOp>(op)
        || isa<OrOp>(op) || isa<XOrOp>(op) || isa<MaxOp>(op) || isa<MinOp>(op)
        || isa<ReduceAndOp>(op) || isa<ReduceOrOp>(op) || isa<ReduceXOrOp>(op))
    {
      pendingBinaryOps.push_back(op);
    } else if (auto assembleOp = dyn_cast<AssembleOp>(op)) {
      assemble = assembleOp;
    }
  });

  bool noneMulF = llvm::none_of(pendingBinaryOps, [](Operation *op) {
    return isa<MulFOp>(op);
  });
  bool allMulF = llvm::all_of(pendingBinaryOps, [](Operation *op) {
    return isa<MulFOp>(op);
  });

  assert((noneMulF || allMulF) &&
      "pendingBinaryOps must contain either all MulFOp or none MulFOp");

  if (allMulF) {
    isMulF = true;
  }

  if (pendingBinaryOps.size() == 1) {
    auto op = pendingBinaryOps[0];
    if (auto maxOp = dyn_cast<MaxOp>(op)) {
      isMax = true;
      gtSignalMap[maxOp.getLhs()] = gtNtk.create_pi();
      gtSignalMap[maxOp.getRhs()] = gtNtk.create_pi();
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      isMin = true;
      ltSignalMap[minOp.getLhs()] = ltNtk.create_pi();
      ltSignalMap[minOp.getRhs()] = ltNtk.create_pi();
    }
  }

  assert((int)isMulF + (int)isMax + (int)isMin <= 1);

  bool progress = true;
  while (progress && !pendingBinaryOps.empty()) {
    progress = false;

    for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
      if (operandsBuilt(*it)) {
        if (auto add = dyn_cast<AddIOp>(*it)) {
          auto lhs = migSignalMap.lookup(add.getLhs());
          auto rhs = migSignalMap.lookup(add.getRhs());
          auto cin = mig.create_pi();
          auto [sum, cout] = mockturtle::full_adder(mig, lhs, rhs, cin);
          mig.create_po(cout);
          migSignalMap[add.getResult()] = sum;
          const int cinIdex = mig.num_pis() - 1;
          assert(!carryMap.contains(cinIdex));
          carryMap[cinIdex] = mig.num_pos() - 1;

          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto andOp = dyn_cast<AndOp>(*it)) {
          auto lhs = migSignalMap.lookup(andOp.getLhs());
          auto rhs = migSignalMap.lookup(andOp.getRhs());
          migSignalMap[andOp.getResult()] = mig.create_and(lhs, rhs);
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto orOp = dyn_cast<OrOp>(*it)) {
          auto lhs = migSignalMap.lookup(orOp.getLhs());
          auto rhs = migSignalMap.lookup(orOp.getRhs());
          migSignalMap[orOp.getResult()] = mig.create_or(lhs, rhs);
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto xorOp = dyn_cast<XOrOp>(*it)) {
          auto lhs = migSignalMap.lookup(xorOp.getLhs());
          auto rhs = migSignalMap.lookup(xorOp.getRhs());
          migSignalMap[xorOp.getResult()] = mig.create_xor(lhs, rhs);
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto redAnd = dyn_cast<ReduceAndOp>(*it)) {
          auto in0 = migSignalMap.lookup(redAnd.getInput());
          migSignalMap[redAnd.getResult()] = mig.create_and(
              in0, mig.create_pi());
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto redOr = dyn_cast<ReduceOrOp>(*it)) {
          auto in0 = migSignalMap.lookup(redOr.getInput());
          migSignalMap[redOr.getResult()] = mig.create_or(
              in0, mig.create_pi());
          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto redXor = dyn_cast<ReduceXOrOp>(*it)) {
          auto in0 = migSignalMap.lookup(redXor.getInput());
          migSignalMap[redXor.getResult()] = mig.create_xor(
              in0, mig.create_pi());
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
        } else if (auto maxOp = dyn_cast<MaxOp>(*it)) {
          const auto lhs = maxOp.getLhs();
          const auto rhs = maxOp.getRhs();
          // GT
          const auto &lhsGTSignal = gtSignalMap[lhs];
          const auto &rhsGTSignal = gtSignalMap[rhs];
          gtNtk.create_po(
              gtNtk.create_and(lhsGTSignal, gtNtk.create_not(rhsGTSignal)));
          
          // Mux
          const auto &lhsSignal = migSignalMap[lhs];
          const auto &rhsSignal = migSignalMap[rhs];
          migSignalMap[maxOp.getResult()] = buildMux2(
              mig, mig.create_pi(), lhsSignal, rhsSignal);

          it = pendingBinaryOps.erase(it);
          progress = true;
        } else if (auto minOp = dyn_cast<MinOp>(*it)) {
          const auto lhs = minOp.getLhs();
          const auto rhs = minOp.getRhs();
          // LT
          const auto &lhsLTSignal = ltSignalMap[lhs];
          const auto &rhsLTSignal = ltSignalMap[rhs];
          ltNtk.create_po(
              ltNtk.create_and(ltNtk.create_not(lhsLTSignal), rhsLTSignal));
          
          // Mux
          const auto &lhsSignal = migSignalMap[lhs];
          const auto &rhsSignal = migSignalMap[rhs];
          migSignalMap[minOp.getResult()] = buildMux2(
              mig, mig.create_pi(), lhsSignal, rhsSignal);

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
  if (isMax) {
    gtNtk = mockturtle::cleanup_dangling(gtNtk);
  }
  if (isMin) {
    ltNtk = mockturtle::cleanup_dangling(ltNtk);
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

MIG::signal NetworkBuilder::buildMux2(MIG &ntk,
                      MIG::signal const &s,
                      MIG::signal const &lhs,
                      MIG::signal const &rhs) {
  auto and1 = ntk.create_and(s, lhs);
  auto and2 = ntk.create_and(ntk.create_not(s), rhs);
  return ntk.create_or(and1, and2);
}

bool NetworkBuilder::operandsBuilt(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!migSignalMap.count(operand))
      return false;
  }
  return true;
}
