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
    if (failed(builder.build())) {
      signalPassFailure();
    }
    const auto mig = builder.getNetwork();

    NetworkBuilder::debugPrint(mig);

    const auto optimized = eggmock::rewrite_mig(mig,
        ambit_rewriter(ambit_compiler_settings{
            .print_program = true, .verbose = true}));

    NetworkBuilder::debugPrint(optimized);
  }
};
} // namespace mlir::bits

std::unique_ptr<mlir::Pass> mlir::bits::createBitsLimeOptimizationPass() {
  return std::make_unique<LimeOptimizationPass>();
}
