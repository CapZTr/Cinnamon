/// Implements the PuD dialect types.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h"

#include <cstdint>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>

#define DEBUG_TYPE "pud-types"

using namespace mlir;
using namespace mlir::pud;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PuDDialect
//===----------------------------------------------------------------------===//

void PuDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.cpp.inc"
        >();
}

Type RowType::parse(AsmParser &parser) {
  StringRef key;

  if (parser.parseLess() || parser.parseKeyword(&key) || parser.parseGreater())
    return Type();
  
  int64_t group;
  if (key == "B") group = 0;
  else if (key == "C") group = 1;
  else if (key == "D") group = 2;
  else {
    parser.emitError(parser.getCurrentLocation(),
        "expected: <B>, <C> or <D>");
    return Type();
  }

  return RowType::get(parser.getContext(), group);
}

void RowType::print(AsmPrinter &printer) const {
  printer << "<";
  switch (getGroup()) {
    case 0: printer << "B"; break;
    case 1: printer << "C"; break;
    case 2: printer << "D"; break;
    default: printer << "Invalid"; break;
  }
  printer << ">";
}
