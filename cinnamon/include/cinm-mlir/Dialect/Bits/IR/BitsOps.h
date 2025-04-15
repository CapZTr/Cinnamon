/// Declaration of the Bits dialect ops.
///
/// @file

#pragma once

#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===- Generated includes -------------------------------------------------===//

#define GET_OP_CLASSES
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h.inc"

//===----------------------------------------------------------------------===//
