#pragma once

#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir::pud {
std::unique_ptr<Pass> createConvertPuDToFuncPass();
}
