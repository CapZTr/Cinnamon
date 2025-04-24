#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>

#include <llvm/Support/SourceMgr.h>
#include <iostream>

using namespace mlir;

int main(int argc, char **argv) {
  MLIRContext context;
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<mlir::bits::BitsDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  if (argc != 4) {
    std::cerr << "Usage: test_bits_codegen <input.mlir> <aig_output.aig> <mig_output.aig>\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  auto buffer = mlir::openInputFile(argv[1]);
  if (!buffer) {
    std::cerr << "Failed to open input file\n";
    return 1;
  }

  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR module\n";
    return 1;
  }

  mlir::bits::NetworkBuilder builder(module.get());
  builder.build(argv[2], argv[3]);

  std::cout << "AIG network written to " << argv[2] << "\n";
  std::cout << "MIG network written to " << argv[3] << "\n";
  return 0;
}
