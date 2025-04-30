#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "ambit.h"
#include "eggmock.h"

#include <iostream>

using AIG = mockturtle::aig_network;
using MIG = mockturtle::mig_network;

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSLIMEOPTIMIZATIONPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::bits {

struct LimeOptimizationPass
    : public ::impl::BitsLimeOptimizationPassBase<LimeOptimizationPass> {
  void runOnOperation() override {
    NetworkBuilder builder(getOperation()->getParentOfType<ModuleOp>());

    auto ntks = builder.build();
    if (!ntks) {
      signalPassFailure();
    }
    auto mig = *ntks;

    // std::cout << "MIG network created.\n";
    // std::cout << "num_gates: " << mig.num_gates() << "\n";
    // std::cout << "num_pis: " << mig.num_pis() << "\n";
    // std::cout << "num_pos: " << mig.num_pos() << "\n";

    ambit_compile_result result = eggmock::send_mig(mig, ambit_compile(ambit_compiler_settings{.print_program = true, .verbose = true}));

    std::cout << "IC:" << result.instruction_count << std::endl;
    std::cout << "t1:" << result.t_runner << std::endl;
    std::cout << "t2:" << result.t_extractor << std::endl;
    std::cout << "t3:" << result.t_compiler << std::endl;
  }
};
} // namespace mlir::bits

std::unique_ptr<mlir::Pass> mlir::bits::createBitsLimeOptimizationPass() {
  return std::make_unique<LimeOptimizationPass>();
}
