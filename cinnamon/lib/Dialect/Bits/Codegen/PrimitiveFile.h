//===- PrimitiveFile.h - row-op primitives defined in an MLIR file --------===//
//
// The networks lime compiles for the row-op primitives are built into
// RowOpLibrary.cpp. --pud-primitives=<file> replaces any of them with one read
// from MLIR, so that a primitive's network can be changed without rebuilding
// cinm-opt. The file is read here, in a translation unit of its own:
// RowOpLibrary.cpp is compiled with RTTI for lime, and LLVM's command-line and
// parsing code cannot be instantiated there against an LLVM built without it.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <mockturtle/networks/mig.hpp>

#include <cstdint>
#include <optional>

namespace mlir::bits {

/// Where lime's cell type list puts each kind of cell: bool, D, DCC, T.
constexpr uint32_t kFileDccCellType = 2;
constexpr uint32_t kFileComputeCellType = 3;

/// A network input or output pinned to a cell, in lime's convention.
struct PrimitiveHint {
  uint32_t index;
  uint32_t cellType;
  uint32_t cellIndex;
};

/// One primitive as the file defines it.
struct PrimitiveDefinition {
  mockturtle::mig_network mig;
  llvm::SmallVector<PrimitiveHint, 2> inputs;
  llvm::SmallVector<PrimitiveHint, 2> outputs;
  /// Whether lime's pre-optimisation may restructure the network first, as it
  /// does the built-in ones. Off, the network is compiled as written.
  bool preoptimize = true;
  /// The command count the compiled sequence must have, if the file says.
  std::optional<unsigned> commands;
};

/// The file's definition of `kind`, or nullptr if there is no file or it does
/// not define that primitive. The file is read on first use.
const PrimitiveDefinition *lookupFilePrimitive(RowOpKind kind);

/// The file named by --pud-primitives, for messages.
llvm::StringRef primitiveFilePath();

/// Names the file for a tool that parses its own arguments rather than through
/// llvm::cl. Only effective before the first lookup.
void setPrimitiveFilePath(llvm::StringRef path);

} // namespace mlir::bits
