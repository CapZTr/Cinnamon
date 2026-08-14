#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir::bits {
std::unique_ptr<Pass> createConvertBitsToPuDPass();
std::unique_ptr<Pass> createConvertBitsToPuDPass(bool doUnroll);

/// Lowers row-level Bits ops by instantiating the row-op templates compiled for
/// the substrate. See ConvertBitsRowsToPuD in BitsPasses.td.
std::unique_ptr<Pass> createConvertBitsRowsToPuDPass();
} // namespace mlir::bits
