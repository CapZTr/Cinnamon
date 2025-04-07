#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include "mockturtle/io/write_aiger.hpp"
#include "mockturtle/algorithms/cleanup.hpp"
#include "mockturtle/generators/arithmetic.hpp"
#include <cstdint>
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
  for (const auto &add : parser.getAdds()) {
    aigOutputs = buildAddInAIG(add);
    migOutputs = buildAddInMIG(add);
  }

  for (auto s : aigOutputs) {
    aig.create_po(s);
  }

  for (auto s : migOutputs) {
    mig.create_po(s);
  }

  // buildOutputs(parser.getAdds().back().result);

  // auto cleanedAig = mockturtle::cleanup_dangling(aig);
  // mockturtle::write_aiger(cleanedAig, filePathAig);

  // auto cleanedMig = mockturtle::cleanup_dangling(mig);
  // mockturtle::write_aiger(cleanedMig, filePathMig);
  mockturtle::write_aiger(aig, filePathAig);
  mockturtle::write_aiger(mig, filePathMig);
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

std::vector<AIG::signal> NetworkBuilder::buildAddInAIG(const AddData &add) {
  auto lhs = add.lhs;
  auto rhs = add.rhs;
  auto result = add.result;
  const int64_t N = lhs.bitWidth * lhs.vectorLength;

  assert(lhs.bitWidth == rhs.bitWidth && lhs.vectorLength == rhs.vectorLength &&
         "NetworkBuilder: Operands must have the same shape");

  auto aigLhsSignal = aigSignalMap.lookup(lhs);
  auto aigRhsSignal = aigSignalMap.lookup(rhs);
  std::vector<AIG::signal> aigSum;
  AIG::signal aigCarry = aig.get_constant(false);

  for (int64_t i = 0; i < N; ++i) {
    auto [as, ac] = mockturtle::full_adder(aig, aigLhsSignal[i], aigRhsSignal[i], aigCarry);
    aigSum.push_back(as);
    aigCarry = ac;
  }
  aigSignalMap[result] = aigSum;
  std::cout << "aigSum size = " << aigSum.size() << "\n";
  std::cout << "AIG gates = " << aig.num_gates() << "\n";

  return aigSum;
}

std::vector<MIG::signal> NetworkBuilder::buildAddInMIG(const AddData &add) {
  auto lhs = add.lhs;
  auto rhs = add.rhs;
  auto result = add.result;
  const int64_t N = lhs.bitWidth * lhs.vectorLength;

  assert(lhs.bitWidth == rhs.bitWidth && lhs.vectorLength == rhs.vectorLength &&
         "NetworkBuilder: Operands must have the same shape");

  auto migLhsSignal = migSignalMap.lookup(lhs);
  auto migRhsSignal = migSignalMap.lookup(rhs);
  std::vector<MIG::signal> migSum;
  MIG::signal migCarry = mig.get_constant(false);

  for (int64_t i = 0; i < N; ++i) {
    auto [ms, mc] = createFullAdderInMIG(mig.create_pi(), mig.create_pi(), migCarry);
    migSum.push_back(ms);
    migCarry = mc;
  }
  migSignalMap[result] = migSum;
  std::cout << "MIG gates = " << mig.num_gates() << "\n";
  std::cout << "migSum size = " << migSum.size() << "\n";

  return migSum;
}

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
  std::cout << "[MIG] MAJ(a,b,cin) → node " << mig.node_to_index(cout.index) << "\n";
  auto maj = mig.create_maj(a, b, mig.create_not(cin));
  std::cout << "[MIG] MAJ(a,b,!cin) → node " << mig.node_to_index(maj.index) << "\n";

  auto sum = mig.create_maj(maj, cin, mig.create_not(cout));
  std::cout << "[MIG] MAJ(maj,cin,!cout) → node " << mig.node_to_index(sum.index) << "\n";

  return {sum, cout};
}
