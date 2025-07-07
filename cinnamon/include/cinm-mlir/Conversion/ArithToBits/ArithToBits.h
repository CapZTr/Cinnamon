#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"

#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Pass/Pass.h>

namespace mlir::bits_frontend {
std::unique_ptr<Pass> createConvertArithToBitsPass();
} // namespace mlir::bits_frontend
