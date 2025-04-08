#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include "mockturtle/io/write_aiger.hpp"
#include "mockturtle/algorithms/cleanup.hpp"
#include "mockturtle/generators/arithmetic.hpp"
#include <cassert>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::bits;

using AIG = mockturtle::aig_network;
using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : parser(module) {
  parser.parse();
}

void NetworkBuilder::build(const std::string &filePathAig, const std::string &filePathMig) {
  for (auto &[val, data] : parser.getInputs()) {
    buildInputs(data);
  }

  std::vector<AIG::signal> aigOutputs;
  std::vector<MIG::signal> migOutputs;
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
    }
  }

  for (const auto &entry : parser.getOutputs()) {
    buildOutputs(entry.second);
  }

  auto cleanedAig = mockturtle::cleanup_dangling(aig);
  mockturtle::write_aiger(cleanedAig, filePathAig);

  auto cleanedMig = mockturtle::cleanup_dangling(mig);
  mockturtle::write_aiger(cleanedMig, filePathMig);
}

void NetworkBuilder::buildInputs(const BitplaneData &data) {
  std::vector<AIG::signal> aigInputs;
  std::vector<MIG::signal> migInputs;
  for (int64_t i = 0; i < data.bitWidth * data.vectorLength; ++i) {
    aigInputs.push_back(aig.create_pi());
    migInputs.push_back(mig.create_pi());
  }
  aigSignalMap[data] = aigInputs;
  migSignalMap[data] = migInputs;
}

void NetworkBuilder::buildAdd(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
  // AIG
  auto aigLhsSignal = aigSignalMap.lookup(lhs);
  auto aigRhsSignal = aigSignalMap.lookup(rhs);
  AIG::signal aigCarry = aig.get_constant(false);

  mockturtle::carry_ripple_adder_inplace(aig, aigLhsSignal, aigRhsSignal, aigCarry);
  aigSignalMap[result] = aigLhsSignal;

  // MIG
  auto migLhsSignal = migSignalMap.lookup(lhs);
  auto migRhsSignal = migSignalMap.lookup(rhs);
  MIG::signal migCarry = mig.get_constant(false);

  mockturtle::carry_ripple_adder_inplace(mig, migLhsSignal, migRhsSignal, migCarry);
  migSignalMap[result] = migLhsSignal;
}

void NetworkBuilder::buildSub(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
  // AIG
  auto aigLhsSignal = aigSignalMap.lookup(lhs);
  auto aigRhsSignal = aigSignalMap.lookup(rhs);
  AIG::signal aigCarry = aig.get_constant(true);

  mockturtle::carry_ripple_subtractor_inplace(aig, aigLhsSignal, aigRhsSignal, aigCarry);
  aigSignalMap[result] = aigLhsSignal;

  // MIG
  auto migLhsSignal = migSignalMap.lookup(lhs);
  auto migRhsSignal = migSignalMap.lookup(rhs);
  MIG::signal migCarry = mig.get_constant(true);

  mockturtle::carry_ripple_subtractor_inplace(mig, migLhsSignal, migRhsSignal, migCarry);
  migSignalMap[result] = migLhsSignal;
}

// void NetworkBuilder::buildAddInAIG(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
//   auto aigLhsSignal = aigSignalMap.lookup(lhs);
//   auto aigRhsSignal = aigSignalMap.lookup(rhs);
//   std::vector<AIG::signal> aigSum;
//   AIG::signal aigCarry = aig.get_constant(false);

//   for (int64_t i = 0; i < lhs.bitWidth * lhs.vectorLength; ++i) {
//     auto [as, ac] = mockturtle::full_adder(aig, aigLhsSignal[i], aigRhsSignal[i], aigCarry);
//     aigSum.push_back(as);
//     aigCarry = ac;
//   }
//   aigSignalMap[result] = aigSum;
// }

// void NetworkBuilder::buildAddInMIG(const BitplaneData &lhs, const BitplaneData &rhs, const BitplaneData &result) {
//   auto migLhsSignal = migSignalMap.lookup(lhs);
//   auto migRhsSignal = migSignalMap.lookup(rhs);
//   std::vector<MIG::signal> migSum;
//   MIG::signal migCarry = mig.get_constant(false);

//   for (int64_t i = 0; i < lhs.bitWidth * lhs.vectorLength; ++i) {
//     auto [ms, mc] = createFullAdderInMIG(migLhsSignal[i], migRhsSignal[i], migCarry);
//     migSum.push_back(ms);
//     migCarry = mc;
//   }
//   migSignalMap[result] = migSum;
// }

void NetworkBuilder::buildOutputs(const BitplaneData &output) {
  auto aigOutputSignal = aigSignalMap.lookup(output);
  for (auto s : aigOutputSignal) {
    aig.create_po(s);
  }

  auto migOutputSignal = migSignalMap.lookup(output);
  for (auto s : migOutputSignal) {
    mig.create_po(s);
  }
}

std::pair<MIG::signal, MIG::signal> NetworkBuilder::createFullAdderInMIG(const MIG::signal a, const MIG::signal b, const MIG::signal cin) {
  auto cout = mig.create_maj(a, b, cin);
  auto maj = mig.create_maj(a, b, mig.create_not(cin));
  auto sum = mig.create_maj(maj, cin, mig.create_not(cout));

  return {sum, cout};
}
