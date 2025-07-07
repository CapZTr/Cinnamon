#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir::bits {
std::unique_ptr<Pass> createConvertBitsToPuDPass();
} // namespace mlir::bits
