/// Implements the PuD dialect types.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinAttributes.h>
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

Type RowAddressType::parse(AsmParser &parser) {
  StringRef groupStr;
  IntegerAttr indexAttr;

  if (parser.parseLess() ||
      parser.parseKeyword(&groupStr) ||
      parser.parseComma() ||
      parser.parseAttribute(indexAttr) ||
      parser.parseGreater()) {
    return Type();
  }

  auto group = symbolizeEnum<RowGroup>(groupStr);
  if (!group)
    return parser.emitError(parser.getNameLoc(), "invalid RowGroup enum value: ") << groupStr, Type();

  return RowAddressType::get(parser.getContext(), *group, indexAttr);
}

void RowAddressType::print(AsmPrinter &printer) const {
  printer << "<" << stringifyEnum(getGroup()) << ", ";
  printer.printAttribute(getIndex());
  printer << ">";
}
