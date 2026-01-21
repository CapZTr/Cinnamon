#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"

#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Pass/Pass.h>

namespace mlir::bits_frontend {
std::unique_ptr<Pass> createConvertLinalgToBitsPass();
} // namespace mlir::bits_frontend
