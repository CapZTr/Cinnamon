//===- rowop-sanity.cpp - print the compiled row-op templates -------------===//
//
// Compiles every row-op primitive through lime and prints the resulting
// instruction templates next to the command counts of the hand-written
// expansions in rowop_trace_player.cc.
//
// This is the audit trail for the cost model: the ILP prices slice-level ops out
// of exactly these numbers, so any divergence from the hand-written expansions
// should be visible here rather than buried in a benchmark result.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"
#include "../../lib/Dialect/Bits/Codegen/PrimitiveFile.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir::bits;

static void printRecipes(PudBackend backend) {
  struct Entry {
    const char *name;
    std::optional<RowOpRecipe> recipe;
    const char *legacy;
  };

  // The legacy column is the closed form the hard-coded cost table used, so a
  // reader can check the recipe reproduces it where the templates agree with the
  // hand-written expansions.
  for (int64_t n : {8, 16, 32}) {
    llvm::outs() << "\nslice-op costs at width " << n << "\n";
    Entry entries[] = {
        {"add_i", getRecipeForAdd(n, false), "8n + 1"},
        {"add_i_full", getRecipeForAdd(n, true), "8n + 2"},
        {"sub_i", getRecipeForSub(n), "9n + 1"},
        {"and", getRecipeForAnd(n, false), "4n"},
        {"and (masked)", getRecipeForAnd(n, true), "4n + 2"},
        {"or", getRecipeForOr(n), "4n"},
        {"xnor", getRecipeForXnor(n), "7n"},
        {"range_scan", getRecipeForRangeScan(n), "4L + 5"},
        {"min/max", getRecipeForMinMax(n), "13n + 2"},
        {"mul_i (n x n)", getRecipeForMul(n, n), "13mn + 4m - 8"},
    };
    for (const Entry &entry : entries) {
      llvm::outs() << "  " << entry.name << ": ";
      if (!entry.recipe) {
        llvm::outs() << "no recipe\n";
        continue;
      }
      auto total = entry.recipe->totalCommands(backend);
      if (!total) {
        llvm::outs() << "not compilable for this backend\n";
        continue;
      }
      llvm::outs() << *total << " commands (legacy closed form: " << entry.legacy
                   << ")\n";
    }
  }
}

int main(int argc, char **argv) {
  // Argument parsing is done by hand rather than with llvm::cl so this tool can
  // link against a BitsCodegen built with RTTI on, which LLVM's own libraries
  // are not.
  std::string backendOpt = "simdram";
  bool recipesOpt = false;
  for (int i = 1; i < argc; i++) {
    llvm::StringRef arg = argv[i];
    if (arg.consume_front("--backend="))
      backendOpt = arg.str();
    else if (arg == "--recipes")
      recipesOpt = true;
    else if (arg.consume_front("--pud-primitives="))
      setPrimitiveFilePath(arg);
    else {
      llvm::errs() << "usage: " << argv[0]
                   << " [--backend=simdram|prada|fcdram] [--recipes]"
                      " [--pud-primitives=<file>]\n";
      return 1;
    }
  }

  PudBackend backend;
  if (backendOpt == "simdram")
    backend = PudBackend::Simdram;
  else if (backendOpt == "prada")
    backend = PudBackend::Prada;
  else if (backendOpt == "fcdram")
    backend = PudBackend::Fcdram;
  else {
    llvm::errs() << "unknown backend '" << backendOpt << "'\n";
    return 1;
  }

  llvm::outs() << RowOpLibrary::get().renderSanityTable(backend);
  if (recipesOpt)
    printRecipes(backend);
  return 0;
}
