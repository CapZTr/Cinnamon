#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <cassert>
#include <cstdint>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/FunctionExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/generators/arithmetic.hpp>

using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : module(module) {}

NetworkBuilder::NetworkBuilder(func::FuncOp func)
    : module(func->getParentOfType<ModuleOp>()), func(func) {}

LogicalResult NetworkBuilder::build() {
  auto isBinaryOp = [](Operation *op) {
    return isa<AddIOp>(op) || isa<MulFOp>(op) || isa<AndOp>(op) ||
           isa<OrOp>(op) || isa<XOrOp>(op) || isa<MaxOp>(op) ||
           isa<MinOp>(op) || isa<MulIOp>(op);
  };

  auto operandsBuiltIn = [](Operation *op,
                            const DenseMap<Value, MIG::signal> &signalMap) {
    return llvm::all_of(op->getOperands(), [&](Value operand) {
      return signalMap.count(operand);
    });
  };

  auto processBinaryOp =
      [&](Operation *op, DenseMap<Value, MIG::signal> &signalMap, MIG &ntk,
          DenseMap<int, int> &carryMapRef,
          llvm::function_ref<void(MulFOp)> onMulF,
          llvm::function_ref<void(MulIOp)> onMulI,
          llvm::function_ref<void(MaxOp)> onMax,
          llvm::function_ref<void(MinOp)> onMin) -> LogicalResult {
    if (auto add = dyn_cast<AddIOp>(op)) {
      const auto lhs = signalMap.lookup(add.getLhs());
      const auto rhs = signalMap.lookup(add.getRhs());
      const auto cin = ntk.create_pi();
      const auto [sum, cout] = mockturtle::full_adder(ntk, lhs, rhs, cin);
      ntk.create_po(cout);
      signalMap[add.getResult()] = sum;
      const int cinIndex = ntk.num_pis() - 1;
      carryMapRef[cinIndex] = ntk.num_pos() - 1;
      return success();
    }

    if (auto andOp = dyn_cast<AndOp>(op)) {
      const auto lhs = signalMap.lookup(andOp.getLhs());
      const auto rhs = signalMap.lookup(andOp.getRhs());
      signalMap[andOp.getResult()] = ntk.create_and(lhs, rhs);
      return success();
    }

    if (auto orOp = dyn_cast<OrOp>(op)) {
      const auto lhs = signalMap.lookup(orOp.getLhs());
      const auto rhs = signalMap.lookup(orOp.getRhs());
      signalMap[orOp.getResult()] = ntk.create_or(lhs, rhs);
      return success();
    }

    if (auto xorOp = dyn_cast<XOrOp>(op)) {
      const auto lhs = signalMap.lookup(xorOp.getLhs());
      const auto rhs = signalMap.lookup(xorOp.getRhs());
      signalMap[xorOp.getResult()] = ntk.create_xor(lhs, rhs);
      return success();
    }

    // if (auto redAnd = dyn_cast<ReduceAndOp>(op)) {
    //   const auto in0 = signalMap.lookup(redAnd.getInput());
    //   signalMap[redAnd.getResult()] = ntk.create_and(in0, ntk.create_pi());
    //   return success();
    // }

    // if (auto redOr = dyn_cast<ReduceOrOp>(op)) {
    //   const auto in0 = signalMap.lookup(redOr.getInput());
    //   signalMap[redOr.getResult()] = ntk.create_or(in0, ntk.create_pi());
    //   return success();
    // }

    // if (auto redXor = dyn_cast<ReduceXOrOp>(op)) {
    //   const auto in0 = signalMap.lookup(redXor.getInput());
    //   signalMap[redXor.getResult()] = ntk.create_xor(in0, ntk.create_pi());
    //   return success();
    // }

    if (auto mul = dyn_cast<MulFOp>(op)) {
      onMulF(mul);
      const auto lhs = signalMap.lookup(mul.getLhs());
      const auto rhs = signalMap.lookup(mul.getRhs());
      const auto cin = ntk.create_pi();
      const auto [sum, cout] = buildAdd(ntk, lhs, rhs, cin);
      (void)cout;
      signalMap[mul.getResult()] = sum;
      const int cinIndex = ntk.num_pis() - 1;
      carryMapRef[cinIndex] = ntk.num_pos() - 1;
      return success();
    }

    if (auto mul = dyn_cast<MulIOp>(op)) {
      onMulI(mul);
      const auto lhs = signalMap.lookup(mul.getLhs());
      const auto rhs = signalMap.lookup(mul.getRhs());
      return success();
    }

    if (auto maxOp = dyn_cast<MaxOp>(op)) {
      onMax(maxOp);
      const auto lhsSignal = signalMap.lookup(maxOp.getLhs());
      const auto rhsSignal = signalMap.lookup(maxOp.getRhs());
      signalMap[maxOp.getResult()] =
          buildMux2(ntk, ntk.create_pi(), lhsSignal, rhsSignal);
      return success();
    }

    if (auto minOp = dyn_cast<MinOp>(op)) {
      onMin(minOp);
      const auto lhsSignal = signalMap.lookup(minOp.getLhs());
      const auto rhsSignal = signalMap.lookup(minOp.getRhs());
      signalMap[minOp.getResult()] =
          buildMux2(ntk, ntk.create_pi(), lhsSignal, rhsSignal);
      return success();
    }

    return failure();
  };

  auto buildPendingOps = [&](SmallVectorImpl<Operation *> &pendingOps,
                             DenseMap<Value, MIG::signal> &signalMap, MIG &ntk,
                             DenseMap<int, int> &carryMapRef,
                             llvm::function_ref<void(MulFOp)> onMulF,
                             llvm::function_ref<void(MulIOp)> onMulI,
                             llvm::function_ref<void(MaxOp)> onMax,
                             llvm::function_ref<void(MinOp)> onMin,
                             bool emitUnsupportedError,
                             bool emitMissingOperandError) -> LogicalResult {
    bool progress = true;
    while (progress && !pendingOps.empty()) {
      progress = false;

      for (auto it = pendingOps.begin(); it != pendingOps.end();) {
        Operation *op = *it;
        if (!operandsBuiltIn(op, signalMap)) {
          ++it;
          continue;
        }

        if (failed(processBinaryOp(op, signalMap, ntk, carryMapRef, onMulF,
                                   onMulI, onMax, onMin))) {
          if (emitUnsupportedError) {
            emitError(op->getLoc(),
                      "NetworkBuilder: Unsupported type of binary op");
          }
          return failure();
        }

        it = pendingOps.erase(it);
        progress = true;
      }
    }

    if (!pendingOps.empty()) {
      if (emitMissingOperandError) {
        llvm::errs() << "NetworkBuilder: Some binary ops could not be parsed "
                        "due to missing operands\n";
      }
      return failure();
    }

    return success();
  };

  auto buildBinaryNetwork = [&](ArrayRef<Operation *> ops,
                                ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                                SubgraphNetwork &out) -> LogicalResult {
    DenseMap<Value, MIG::signal> localSignalMap;
    SmallVector<Operation *> pendingBinaryOps;

    for (Value input : inputs) {
      localSignalMap[input] = out.mig.create_pi();
    }

    for (Operation *op : ops) {
      if (isBinaryOp(op)) {
        pendingBinaryOps.push_back(op);
      }
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

    if (failed(buildPendingOps(
            pendingBinaryOps, localSignalMap, out.mig, out.carryMap,
            [](MulFOp) {}, [](MulIOp) {}, [](MaxOp) {}, [](MinOp) {},
            /*emitUnsupportedError=*/false,
            /*emitMissingOperandError=*/false))) {
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

  walkTarget([&](Operation *op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      auto slice = transpose.getOutput();
      auto elemType = cast<RankedTensorType>(transpose.getInput().getType())
                          .getElementType();
      migSignalMap[slice] = mig.create_pi();
      if (isa<FloatType>(elemType)) {
        mulFSignSignalMap[slice] = mulFSignNtk.create_pi();
      }
      inputSlices.push_back(slice);
    } else if (isBinaryOp(op)) {
      pendingBinaryOps.push_back(op);
    } else if (auto assembleOp = dyn_cast<AssembleOp>(op)) {
      assemble = assembleOp;
    }
  });

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

  if (failed(buildPendingOps(
          pendingBinaryOps, migSignalMap, mig, carryMap,
          [&](MulFOp mul) {
            const auto lhsSign = mulFSignSignalMap.lookup(mul.getLhs());
            const auto rhsSign = mulFSignSignalMap.lookup(mul.getRhs());
            mulFSignSignalMap[mul.getResult()] =
                mulFSignNtk.create_xor(lhsSign, rhsSign);
          },
          [&](MulIOp mul) {},
          [&](MaxOp maxOp) {
            const auto lhs = maxOp.getLhs();
            const auto rhs = maxOp.getRhs();
            const auto lhsGTSignal = gtSignalMap.lookup(lhs);
            const auto rhsGTSignal = gtSignalMap.lookup(rhs);
            gtNtk.create_po(
                gtNtk.create_and(lhsGTSignal, gtNtk.create_not(rhsGTSignal)));
          },
          [&](MinOp minOp) {
            const auto lhs = minOp.getLhs();
            const auto rhs = minOp.getRhs();
            const auto lhsLTSignal = ltSignalMap.lookup(lhs);
            const auto rhsLTSignal = ltSignalMap.lookup(rhs);
            ltNtk.create_po(
                ltNtk.create_and(ltNtk.create_not(lhsLTSignal), rhsLTSignal));
          },
          /*emitUnsupportedError=*/true,
          /*emitMissingOperandError=*/true))) {
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
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(),
                                                         "bits.value_id")) {
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

std::pair<MIG::signal, MIG::signal>
NetworkBuilder::buildAdd(MIG &ntk, MIG::signal const &lhs,
                         MIG::signal const &rhs, MIG::signal const &cin) {
  auto cout = ntk.create_maj(lhs, rhs, cin);
  ntk.create_po(cout);
  auto maj = ntk.create_maj(lhs, rhs, ntk.create_not(cin));
  auto sum = ntk.create_maj(maj, cin, ntk.create_not(cout));
  return {sum, cout};
}

MIG::signal NetworkBuilder::buildMux2(MIG &ntk, MIG::signal const &s,
                                      MIG::signal const &lhs,
                                      MIG::signal const &rhs) {
  auto and1 = ntk.create_and(s, lhs);
  auto and2 = ntk.create_and(ntk.create_not(s), rhs);
  return ntk.create_or(and1, and2);
}
