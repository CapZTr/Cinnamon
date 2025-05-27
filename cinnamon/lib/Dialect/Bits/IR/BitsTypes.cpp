/// Implements the Bits dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <llvm/ADT/TypeSwitch.h>

#define DEBUG_TYPE "bits-types"

using namespace mlir;
using namespace mlir::bits;

//===- Generated implementation -------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.cpp.inc"

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BitsDialect
//===----------------------------------------------------------------------===//

void BitsDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.cpp.inc"
        >();
}

Type RowAddressType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  int64_t bankID, subarrayID, rowID;
  if (parser.parseInteger(bankID) || parser.parseComma() ||
      parser.parseInteger(subarrayID) || parser.parseComma() ||
      parser.parseInteger(rowID) || parser.parseGreater()) {
    return Type();
  }

  return RowAddressType::get(parser.getContext(), bankID, subarrayID, rowID);
}

void RowAddressType::print(AsmPrinter &printer) const {
  printer << "<" << getBankID() << ", "
      << getSubarrayID() << ", "
      << getRowID() << ">";
}

Type SliceType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t> shape;
  Type rowAddrType;

  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/false, /*withTrailingX=*/false) ||
      parser.parseGreater() ||
      parser.parseComma() ||
      parser.parseType(rowAddrType) ||
      !mlir::isa<RowAddressType>(rowAddrType) ||
      parser.parseGreater()) {
    return Type();
  }

  auto firstRow = mlir::cast<RowAddressType>(rowAddrType);

  return SliceType::get(parser.getContext(), shape.front(), shape.back(), firstRow);
}

void SliceType::print(AsmPrinter &printer) const {
  printer << "<" 
      << "<" << getBitWidth() << "x" << getVectorLength() << ">"
      << ", " << getFirstRowAddr() << ">";
}
