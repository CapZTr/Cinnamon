/// Implements the Bits dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include <cstdint>
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

Type SliceType::parse(AsmParser &parser) {
  SmallVector<int64_t, 2> shape;

  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/false, /*withTrailingX=*/false) ||
      parser.parseGreater()) {
    return Type();
  }

  return SliceType::get(parser.getContext(), shape.front(), shape.back());
}

void SliceType::print(AsmPrinter &printer) const {
  printer << "<" << getBitWidth() << "x" << getVectorLength() << ">";
}

Type CubeType::parse(AsmParser &parser) {
  SmallVector<int64_t, 3> shape;

  if (parser.parseLess() ||
      parser.parseDimensionList(shape, false, false) ||
      parser.parseGreater()) {
    return  Type();
  }

  return CubeType::get(parser.getContext(), shape[0], shape[1], shape[2]);
}

void CubeType::print(AsmPrinter &printer) const {
  printer << "<" << getBitWidth() << "x" << getVectorLength() << "x" << getHeight() << ">";
}
