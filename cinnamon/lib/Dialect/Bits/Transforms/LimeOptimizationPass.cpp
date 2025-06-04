#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include <cstdint>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <iostream>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/write_dot.hpp>

#include "ambit.h"

using MIG = mockturtle::mig_network;

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSLIMEOPTIMIZATIONPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bits {

struct RowAddress {
  int64_t bank;
  int64_t subarray;
  int64_t row;
};

class AddressAllocator {
public:
  AddressAllocator() = default;

  RowAddress allocate(int64_t numRows) {
    if (currentRow + numRows > MAX_ROW) {
      currentRow = 0;
      ++currentSubarray;
      if (currentSubarray > MAX_SUBARRAY) {
        currentSubarray = 0;
        ++currentBank;
        if (currentBank > MAX_BANK)
          llvm::report_fatal_error(
              "AddressAllocator: DRAM address space exhausted");
      }
    }
    RowAddress addr = {currentBank, currentSubarray, currentRow};
    currentRow += numRows;
    return addr;
  }

private:
  const int64_t MAX_BANK = 15;
  const int64_t MAX_SUBARRAY = 31;
  const int64_t MAX_ROW = 1005;
  int64_t currentBank = 0;
  int64_t currentSubarray = 0;
  int64_t currentRow = 0;
};

struct GlobalAddressAllocator {
  static AddressAllocator &get() {
    static AddressAllocator allocator;
    return allocator;
  }
};

struct InputCache {
  static llvm::DenseMap<Value, Value> &get() {
    static llvm::DenseMap<Value, Value> cache;
    return cache;
  }
};

struct SliceCache {
  static llvm::DenseMap<Operation*, Value> &get() {
    static llvm::DenseMap<Operation*, Value> cache;
    return cache;
  }
};

struct LimeOptimizationPass
    : public ::impl::BitsLimeOptimizationPassBase<LimeOptimizationPass> {
  void runOnOperation() override {
    NetworkBuilder builder(getOperation()->getParentOfType<ModuleOp>());
    if (failed(builder.build())) {
      signalPassFailure();
    }
    auto mig = builder.getNetwork();

    std::cout << builder.getInputSlices().size() << "\n";

    NetworkBuilder::debugPrint(mig);

    std::cout << " ===== Generated Network ===== \n";
    mockturtle::write_dot(mig, std::cout);

    const auto settings = ambit_compiler_settings{
        .print_program = false,
        .verbose = false,
        .preoptimize = true,
        .rewrite = false,
    };

    ProgramString program_str;
    auto [optimized, result] = ambit_rewrite(settings, mig, program_str);
    std::cout << "Generated program:\n" << program_str.str() << "\n";

    NetworkBuilder::debugPrint(optimized);

    std::cout << " ===== Optimized Network ===== \n";
    mockturtle::write_dot(optimized, std::cout);
  }
};
} // namespace mlir::bits

std::unique_ptr<mlir::Pass> mlir::bits::createBitsLimeOptimizationPass() {
  return std::make_unique<LimeOptimizationPass>();
}
