/// Implements the PuD dialect types.
///
/// @file

#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h"

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

Type BitwiseRowAddressType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  int64_t bankID, subarrayID, rowID;
  if (parser.parseInteger(bankID) || parser.parseComma() ||
      parser.parseInteger(subarrayID) || parser.parseComma() ||
      parser.parseInteger(rowID) || parser.parseGreater()) {
    return Type();
  }

  return BitwiseRowAddressType::get(parser.getContext(), bankID, subarrayID, rowID);
}

void BitwiseRowAddressType::print(AsmPrinter &printer) const {
  printer << "<"
      << getBankID() << ", "
      << getSubarrayID() << ", "
      << "B" << getRowID() << ">";
}

Type ControlRowAddressType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  int64_t bankID, subarrayID, rowID;
  if (parser.parseInteger(bankID) || parser.parseComma() ||
      parser.parseInteger(subarrayID) || parser.parseComma() ||
      parser.parseInteger(rowID) || parser.parseGreater()) {
    return Type();
  }

  return ControlRowAddressType::get(parser.getContext(), bankID, subarrayID, rowID == 1);
}

void ControlRowAddressType::print(AsmPrinter &printer) const {
  int rowID = getRowID() ? 1 : 0;
  printer << "<"
      << getBankID() << ", "
      << getSubarrayID() << ", "
      << "C" << rowID << ">";
}

Type DataRowAddressType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  int64_t bankID, subarrayID, rowID;
  if (parser.parseInteger(bankID) || parser.parseComma() ||
      parser.parseInteger(subarrayID) || parser.parseComma() ||
      parser.parseInteger(rowID) || parser.parseGreater()) {
    return Type();
  }

  return DataRowAddressType::get(parser.getContext(), bankID, subarrayID, rowID);
}

void DataRowAddressType::print(AsmPrinter &printer) const {
  printer << "<"
      << getBankID() << ", "
      << getSubarrayID() << ", "
      << "D" << getRowID() << ">";
}
