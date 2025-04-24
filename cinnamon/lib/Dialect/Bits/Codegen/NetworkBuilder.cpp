#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include <mockturtle/io/write_aiger.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/generators/arithmetic.hpp>
#include <cassert>
#include <llvm/ADT/STLExtras.h>

using namespace mlir;
using namespace mlir::bits;

using AIG = mockturtle::aig_network;
using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : parser(module) {}

LogicalResult NetworkBuilder::build(const std::string &filePathAig, const std::string &filePathMig) {
  if (failed(parser.parse())) {
    llvm::errs() << "NetworkBuilder: Failed to parse the module.";
    return failure();
  }
  for (auto &[val, data] : parser.getInputs()) {
    aigSignalMap[data] = aig.create_pi();
    migSignalMap[data] = mig.create_pi();
  }

  const auto adds = parser.getAdds();
  const auto subs = parser.getSubs();
  for (const auto &op : parser.getBinaryOps()) {
    auto lhs = op.lhs;
    auto rhs = op.rhs;
    auto result = op.result;

    assert(lhs.bitWidth == rhs.bitWidth && lhs.vectorLength == rhs.vectorLength &&
           "NetworkBuilder: Operands must have the same shape");

    if (llvm::is_contained(adds, op)) {
      buildAdd(lhs, rhs, result);
    } else if (llvm::is_contained(subs, op)) {
      buildSub(lhs, rhs, result);
    } else {
      llvm::errs() << "NetworkBuilder: Unknown type of binary operation.\n";
      return failure();
    }
  }

  for (const auto &entry : parser.getOutputs()) {
    auto data = entry.second;
    aig.create_po(aigSignalMap.lookup(data));
    mig.create_po(migSignalMap.lookup(data));
  }

  auto cleanedAig = mockturtle::cleanup_dangling(aig);
  mockturtle::write_aiger(cleanedAig, filePathAig);

  auto cleanedMig = mockturtle::cleanup_dangling(mig);
  mockturtle::write_aiger(cleanedMig, filePathMig);

  return success();
}

void NetworkBuilder::buildAdd(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
  // AIG
  auto aigLhsSignal = aigSignalMap.lookup(lhs);
  auto aigRhsSignal = aigSignalMap.lookup(rhs);
  AIG::signal aigCarry = aig.get_constant(false);

  auto [as, ac] = mockturtle::full_adder(aig, aigLhsSignal, aigRhsSignal, aigCarry);
  aigSignalMap[result] = as;

  // MIG
  auto migLhsSignal = migSignalMap.lookup(lhs);
  auto migRhsSignal = migSignalMap.lookup(rhs);
  MIG::signal migCarry = mig.get_constant(false);

  auto [ms, mc] = mockturtle::full_adder(mig, migLhsSignal, migRhsSignal, migCarry);

  migSignalMap[result] = ms;
}

void NetworkBuilder::buildSub(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
  // AIG
  auto aigLhsSignal = aigSignalMap.lookup(lhs);
  auto aigRhsSignal = aigSignalMap.lookup(rhs);
  AIG::signal aigCarry = aig.get_constant(true);
  std::vector<AIG::signal> al = {aigLhsSignal};
  std::vector<AIG::signal> ar = {aigRhsSignal};
  mockturtle::carry_ripple_subtractor_inplace(aig, al, ar, aigCarry);

  aigSignalMap[result] = al[0];

  // MIG
  auto migLhsSignal = migSignalMap.lookup(lhs);
  auto migRhsSignal = migSignalMap.lookup(rhs);
  MIG::signal migCarry = mig.get_constant(true);
  std::vector<MIG::signal> ml = {migLhsSignal};
  std::vector<MIG::signal> mr = {migRhsSignal};
  mockturtle::carry_ripple_subtractor_inplace(mig, ml, mr, migCarry);

  migSignalMap[result] = ml[0];
}
