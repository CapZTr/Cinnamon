/// Implements the Bits dialect types.
///
/// @file

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/TypeSwitch.h"
#include <cstdint>
#include <mlir/Analysis/Presburger/IntegerRelation.h>
#include <mlir/IR/Types.h>

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

Type mlir::bits::SliceType::parse(mlir::AsmParser &parser) {
  SmallVector<int64_t> shape;

  if (parser.parseLess() ||
      parser.parseDimensionList(shape, /*allowDynamic=*/false, /*withTrailingX=*/false) ||
      parser.parseGreater()) {
    return Type();
  }

  return bits::SliceType::get(parser.getContext(), shape.front(), shape.back());
}

void mlir::bits::SliceType::print(mlir::AsmPrinter &printer) const {
  printer << "<" << getBitWidth() << "x" << getVectorLength() << ">";
}
