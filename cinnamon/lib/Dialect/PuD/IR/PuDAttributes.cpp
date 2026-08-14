/// Implements the PuD dialect attributes.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::pud;

//===- Generated implementation -------------------------------------------===//

#include "cinm-mlir/Dialect/PuD/IR/PuDEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.cpp.inc"

//===----------------------------------------------------------------------===//

void PuDDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.cpp.inc"
      >();
}
