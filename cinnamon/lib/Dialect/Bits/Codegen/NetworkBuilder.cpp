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
#include <cstdint>
#include <mockturtle/algorithms/cleanup.hpp>


using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : module(module) {}

NetworkBuilder::NetworkBuilder(func::FuncOp func)
    : module(func->getParentOfType<ModuleOp>()), func(func) {}

LogicalResult NetworkBuilder::build() {

  auto buildBinaryNetwork = [&](ArrayRef<Operation *> ops,
                                ArrayRef<Value> inputs,
                                ArrayRef<Value> outputs,
                                SubgraphNetwork &out) -> LogicalResult {
    DenseMap<Value, MIG::signal> localSignalMap;
    SmallVector<Operation *> pendingBinaryOps;

    for (Value input : inputs) {
      localSignalMap[input] = out.mig.create_pi();
    }

    for (Operation *op : ops) {
      if (isa<AddIOp>(op) || isa<MulFOp>(op) || isa<AndOp>(op)
          || isa<OrOp>(op) || isa<XOrOp>(op) || isa<MaxOp>(op)
          || isa<MinOp>(op) || isa<ReduceAndOp>(op) || isa<ReduceOrOp>(op)
          || isa<ReduceXOrOp>(op)) {
        pendingBinaryOps.push_back(op);
      }
    }

    bool noneMulF = llvm::none_of(pendingBinaryOps, [](Operation *op) {
      return isa<MulFOp>(op);
    });
    bool allMulF = llvm::all_of(pendingBinaryOps, [](Operation *op) {
      return isa<MulFOp>(op);
    });

    if (!(noneMulF || allMulF)) {
      return failure();
    }

    if (allMulF) {
      out.isMulF = true;
    }

    if (pendingBinaryOps.size() == 1) {
      auto op = pendingBinaryOps[0];
      if (auto maxOp = dyn_cast<MaxOp>(op)) {
        out.isMax = true;
        out.gtNtk.create_pi();
        out.gtNtk.create_pi();
      } else if (auto minOp = dyn_cast<MinOp>(op)) {
        out.isMin = true;
        out.ltNtk.create_pi();
        out.ltNtk.create_pi();
      }
    }

    bool progress = true;
    while (progress && !pendingBinaryOps.empty()) {
      progress = false;

      for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
        Operation *op = *it;
        bool operandsReady = true;
        for (Value operand : op->getOperands()) {
          if (!localSignalMap.count(operand)) {
            operandsReady = false;
            break;
          }
        }
        if (!operandsReady) {
          ++it;
          continue;
        }

        if (auto add = dyn_cast<AddIOp>(op)) {
          auto lhs = localSignalMap.lookup(add.getLhs());
          auto rhs = localSignalMap.lookup(add.getRhs());
          auto cin = out.mig.create_pi();
          auto [sum, cout] = mockturtle::full_adder(out.mig, lhs, rhs, cin);
          out.mig.create_po(cout);
          localSignalMap[add.getResult()] = sum;
          const int cinIndex = out.mig.num_pis() - 1;
          out.carryMap[cinIndex] = out.mig.num_pos() - 1;
        } else if (auto andOp = dyn_cast<AndOp>(op)) {
          auto lhs = localSignalMap.lookup(andOp.getLhs());
          auto rhs = localSignalMap.lookup(andOp.getRhs());
          localSignalMap[andOp.getResult()] = out.mig.create_and(lhs, rhs);
        } else if (auto orOp = dyn_cast<OrOp>(op)) {
          auto lhs = localSignalMap.lookup(orOp.getLhs());
          auto rhs = localSignalMap.lookup(orOp.getRhs());
          localSignalMap[orOp.getResult()] = out.mig.create_or(lhs, rhs);
        } else if (auto xorOp = dyn_cast<XOrOp>(op)) {
          auto lhs = localSignalMap.lookup(xorOp.getLhs());
          auto rhs = localSignalMap.lookup(xorOp.getRhs());
          localSignalMap[xorOp.getResult()] = out.mig.create_xor(lhs, rhs);
        } else if (auto redAnd = dyn_cast<ReduceAndOp>(op)) {
          auto in0 = localSignalMap.lookup(redAnd.getInput());
          localSignalMap[redAnd.getResult()] = out.mig.create_and(
              in0, out.mig.create_pi());
        } else if (auto redOr = dyn_cast<ReduceOrOp>(op)) {
          auto in0 = localSignalMap.lookup(redOr.getInput());
          localSignalMap[redOr.getResult()] = out.mig.create_or(
              in0, out.mig.create_pi());
        } else if (auto redXor = dyn_cast<ReduceXOrOp>(op)) {
          auto in0 = localSignalMap.lookup(redXor.getInput());
          localSignalMap[redXor.getResult()] = out.mig.create_xor(
              in0, out.mig.create_pi());
        } else if (auto mul = dyn_cast<MulFOp>(op)) {
          auto lhs = localSignalMap.lookup(mul.getLhs());
          auto rhs = localSignalMap.lookup(mul.getRhs());
          auto cin = out.mig.create_pi();
          auto [sum, cout] = buildAdd(out.mig, lhs, rhs, cin);
          localSignalMap[mul.getResult()] = sum;
          const int cinIndex = out.mig.num_pis() - 1;
          out.carryMap[cinIndex] = out.mig.num_pos() - 1;
        } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
          const auto &lhsSignal = localSignalMap.lookup(maxOp.getLhs());
          const auto &rhsSignal = localSignalMap.lookup(maxOp.getRhs());
          localSignalMap[maxOp.getResult()] = buildMux2(
              out.mig, out.mig.create_pi(), lhsSignal, rhsSignal);
        } else if (auto minOp = dyn_cast<MinOp>(op)) {
          const auto &lhsSignal = localSignalMap.lookup(minOp.getLhs());
          const auto &rhsSignal = localSignalMap.lookup(minOp.getRhs());
          localSignalMap[minOp.getResult()] = buildMux2(
              out.mig, out.mig.create_pi(), lhsSignal, rhsSignal);
        } else {
          return failure();
        }

        it = pendingBinaryOps.erase(it);
        progress = true;
      }
    }

    if (!pendingBinaryOps.empty()) {
      return failure();
    }

    for (Value output : outputs) {
      auto it = localSignalMap.find(output);
      if (it == localSignalMap.end()) {
        localSignalMap[output] = out.mig.create_pi();
        it = localSignalMap.find(output);
      }
      out.mig.create_po(it->second);
    }

    out.mig = mockturtle::cleanup_dangling(out.mig);
    return success();
  };

  SmallVector<Operation *> pendingBinaryOps;
  AssembleOp assemble;

  // module.walk([&](Operation* op) {
  auto walkTarget = [&](auto &&callback) {
    if (func) {
      func.walk(callback);
      return;
    }
    module.walk(callback);
  };

  walkTarget([&](Operation* op) {
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

  if (func && func->hasAttr("bits.subgraphs")) {
    DenseMap<int64_t, Operation *> nodeIdToOp;
    DenseMap<int64_t, Value> valueIdToValue;

    func.walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<IntegerAttr>("bits.node_id")) {
        nodeIdToOp[attr.getInt()] = op;
      }
      if (auto valuesAttr = op->getAttrOfType<ArrayAttr>("bits.value_ids")) {
        for (auto [idx, valueAttr] :
             llvm::enumerate(valuesAttr.getAsRange<IntegerAttr>())) {
          valueIdToValue[valueAttr.getInt()] = op->getResult(idx);
        }
      }
    });
    for (BlockArgument arg : func.getArguments()) {
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              arg.getArgNumber(), "bits.value_id")) {
        valueIdToValue[attr.getInt()] = arg;
      }
    }

    auto subgraphsAttr = func->getAttrOfType<ArrayAttr>("bits.subgraphs");
    subgraphNetworks.clear();
    subgraphNetworks.reserve(subgraphsAttr.size());

    for (Attribute attr : subgraphsAttr) {
      auto dict = llvm::dyn_cast<DictionaryAttr>(attr);
      if (!dict) {
        continue;
      }
      SubgraphNetwork subgraph;
      if (auto bankAttr = dict.getAs<IntegerAttr>("bank_id")) {
        subgraph.bankId = bankAttr.getInt();
      }
      auto nodesAttr = dict.getAs<ArrayAttr>("nodes");
      auto inputsAttr = dict.getAs<ArrayAttr>("inputs");
      auto outputsAttr = dict.getAs<ArrayAttr>("outputs");

      SmallVector<Operation *> ops;
      if (nodesAttr) {
        for (auto nodeAttr : nodesAttr.getAsRange<IntegerAttr>()) {
          subgraph.nodeIds.push_back(nodeAttr.getInt());
          if (auto it = nodeIdToOp.find(nodeAttr.getInt());
              it != nodeIdToOp.end()) {
            ops.push_back(it->second);
          }
        }
      }

      SmallVector<Value> inputs;
      if (inputsAttr) {
        for (auto inputAttr : inputsAttr.getAsRange<IntegerAttr>()) {
          subgraph.inputValueIds.push_back(inputAttr.getInt());
          if (auto it = valueIdToValue.find(inputAttr.getInt());
              it != valueIdToValue.end()) {
            inputs.push_back(it->second);
          }
        }
      }

      SmallVector<Value> outputs;
      if (outputsAttr) {
        for (auto outputAttr : outputsAttr.getAsRange<IntegerAttr>()) {
          subgraph.outputValueIds.push_back(outputAttr.getInt());
          if (auto it = valueIdToValue.find(outputAttr.getInt());
              it != valueIdToValue.end()) {
            outputs.push_back(it->second);
          }
        }
      }

      if (failed(buildBinaryNetwork(ops, inputs, outputs, subgraph))) {
        return failure();
      }

      subgraphNetworks.push_back(std::move(subgraph));
    }

    DenseMap<int64_t, int64_t> valueProducerBank;
    for (const auto &subgraph : subgraphNetworks) {
      for (int64_t valueId : subgraph.outputValueIds) {
        valueProducerBank[valueId] = subgraph.bankId;
      }
    }

    DenseMap<int64_t, SmallVector<int64_t>> depMap;
    for (const auto &subgraph : subgraphNetworks) {
      for (int64_t valueId : subgraph.inputValueIds) {
        auto it = valueProducerBank.find(valueId);
        if (it == valueProducerBank.end() || it->second == subgraph.bankId) {
          continue;
        }
        int64_t key = (it->second << 32) | (subgraph.bankId & 0xffffffff);
        depMap[key].push_back(valueId);
      }
    }

    subgraphDependencies.clear();
    for (auto &entry : depMap) {
      SubgraphDependency dep;
      dep.producerBankId = entry.first >> 32;
      dep.consumerBankId = entry.first & 0xffffffff;
      dep.valueIds = entry.second;
      subgraphDependencies.push_back(std::move(dep));
    }
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
