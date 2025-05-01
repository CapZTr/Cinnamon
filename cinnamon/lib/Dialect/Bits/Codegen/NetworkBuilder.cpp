#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/generators/arithmetic.hpp>
#include <cassert>
#include <llvm/ADT/STLExtras.h>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::bits;

using MIG = mockturtle::mig_network;

NetworkBuilder::NetworkBuilder(ModuleOp module) : parser(module) {}

std::optional<MIG> NetworkBuilder::build() {
  if (failed(parser.parse())) {
    llvm::errs() << "NetworkBuilder: Failed to parse the module";
    return std::nullopt;
  }

  for (auto &[val, data] : parser.getInputs()) {
    migSignalMap[val] = mig.create_pi();
  }

  const auto adds = parser.getAdds();
  for (const auto &op : parser.getBinaryOps()) {
    auto lhs = op.lhs;
    auto rhs = op.rhs;
    auto result = op.result;

    assert(lhs.bitWidth == rhs.bitWidth && lhs.vectorLength == rhs.vectorLength
        && "NetworkBuilder: Operands must have the same shape");

    if (llvm::is_contained(adds, op)) {
      buildAdd(lhs, rhs, result);
    } else {
      llvm::errs() << "NetworkBuilder: Unknown type of binary operation.\n";
      return std::nullopt;
    }
  }

  for (const auto &entry : parser.getOutputs()) {
    // auto data = entry.second;
    mig.create_po(migSignalMap.lookup(entry.first));
  }

  auto cleanedMig = mockturtle::cleanup_dangling(mig);

  debugPrintMIGSignals();

  return mig;
}

void NetworkBuilder::buildAdd(const BitplaneData &lhs,
                              const BitplaneData &rhs,
                              const BitplaneData &result) {
  auto migLhsSignal = migSignalMap.lookup(lhs.slice);
  auto migRhsSignal = migSignalMap.lookup(rhs.slice);
  MIG::signal migCarry = mig.get_constant(false);

  auto [ms, mc] = createFullAdderInMIG(migLhsSignal, migRhsSignal, migCarry);

  migSignalMap[result.slice] = ms;
}

std::pair<MIG::signal, MIG::signal> NetworkBuilder::createFullAdderInMIG(
    const MIG::signal a, const MIG::signal b, const MIG::signal cin) {
  auto cout = mig.create_maj(a, b, cin);
  auto maj = mig.create_maj(a, b, mig.create_not(cin));
  auto sum = mig.create_maj(maj, cin, mig.create_not(cout));

  return {sum, cout};
}

void NetworkBuilder::debugPrintMIGSignals(llvm::raw_ostream &os) {
  os << "\n=== MIG Network Debug Info ===\n";
  os << "Primary Inputs (PIs): " << mig.num_pis() << "\n";
  os << "Primary Outputs (POs): " << mig.num_pos() << "\n";
  os << "Gates: " << mig.num_gates() << "\n";
  os << "Total Nodes: " << mig.size() << "\n\n";

  os << "=== Input Signal Map ===\n";
  for (auto &[val, data] : parser.getInputs()) {
    auto sig = migSignalMap.lookup(data.slice);
    os << "  Input " << val << " (Bitplane " << data.toString() 
       << ") -> Node " << mig.get_node(sig) 
       << (mig.is_complemented(sig) ? " (complemented)" : "") << "\n";
  }

  os << "\n=== Node Details ===\n";
  mig.foreach_node([&](auto node) {
    if (mig.is_constant(node) || mig.is_pi(node)) return;

    os << "  Node " << node << ": ";
    
    if (mig.is_maj(node)) {
      os << "MAJ( ";
      mig.foreach_fanin(node, [&](auto const& fanin, auto i) {
        if (i > 0) os << ", ";
        os << mig.get_node(fanin) 
           << (mig.is_complemented(fanin) ? "'" : "");
      });
      os << " )";
    }
    os << "\n";
  });

  os << "\n=== Output Signal Map ===\n";
  for (const auto &entry : parser.getOutputs()) {
    auto val = entry.first;
    auto data = entry.second;
    auto sig = migSignalMap.lookup(data.slice);
    os << "  Output " << val << " (Bitplane " << data.toString() 
       << ") -> Node " << mig.get_node(sig)
       << (mig.is_complemented(sig) ? " (complemented)" : "") << "\n";
  }

  os << "===========================\n\n";
}
