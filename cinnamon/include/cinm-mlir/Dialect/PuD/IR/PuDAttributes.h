/// Declaration of the PuD dialect attributes.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"
#include "mlir/IR/Attributes.h"

//===- Generated includes -------------------------------------------------===//

#include "cinm-mlir/Dialect/PuD/IR/PuDEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h.inc"

//===----------------------------------------------------------------------===//

namespace mlir::pud {

/// How many rows at the front of a bank the substrate keeps for computation.
///
/// They are named rather than numbered, and the enumeration is what says how
/// many there are. Two sides need the answer -- the lowering that puts the data
/// rows after them, and the check that the whole space fits in a subarray -- so
/// it is asked once here rather than restated.
inline int64_t controlRowCount() {
  return static_cast<int64_t>(getMaxEnumValForSimdramCtrlRow()) + 1;
}

} // namespace mlir::pud
