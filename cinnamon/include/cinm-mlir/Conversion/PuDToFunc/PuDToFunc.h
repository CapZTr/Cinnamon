#pragma once

#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir::pud {
std::unique_ptr<Pass> createConvertPuDToFuncPass();

/// Lowers PuD row ops to guest runtime calls, keeping the loop structure.
std::unique_ptr<Pass> createConvertPuDRowsToFuncPass();
}
