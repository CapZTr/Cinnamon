#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Conversion/BitsPasses.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#include <cstdint>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tuple>

using namespace mlir;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/BitsPasses.h.inc>

namespace {

struct RowAddress {
  int64_t bank;
  int64_t subarray;
  int64_t row;
};

class AddressAllocator {
public:
  AddressAllocator() = default;

  RowAddress allocate(int64_t numRows) {
    if (currentRow + numRows > MAX_ROW) {
      currentRow = 0;
      ++currentSubarray;
      if (currentSubarray > MAX_SUBARRAY) {
        currentSubarray = 0;
        ++currentBank;
        if (currentBank > MAX_BANK)
          llvm::report_fatal_error(
              "AddressAllocator: DRAM address space exhausted");
      }
    }
    RowAddress addr = {currentBank, currentSubarray, currentRow};
    currentRow += numRows;
    return addr;
  }

private:
  const int64_t MAX_BANK = 15;
  const int64_t MAX_SUBARRAY = 31;
  const int64_t MAX_ROW = 1005;
  int64_t currentBank = 0;
  int64_t currentSubarray = 0;
  int64_t currentRow = 0;
};

struct GlobalAddressAllocator {
  static AddressAllocator &get() {
    static AddressAllocator allocator;
    return allocator;
  }
};

struct InputCache {
  static llvm::DenseMap<Value, Value> &get() {
    static llvm::DenseMap<Value, Value> cache;
    return cache;
  }
};

struct SliceCache {
  static llvm::DenseMap<Operation *, Value> &get() {
    static llvm::DenseMap<Operation *, Value> cache;
    return cache;
  }
};

} // namespace
