#include "cinm-mlir/Dialect/PuD/Codegen/AddressAllocator.h"

#include <cassert>

using namespace mlir::pud;

RowAddress AddressAllocator::bankAddress(int64_t flatBank) const {
  assert(geometry.banksPerChannel > 0 && "a channel holds at least one bank");
  RowAddress addr;
  // The mapping numbers banks flatly across the whole memory; a channel holds a
  // fixed number of them, so the index splits on that boundary.
  addr.channel = flatBank / geometry.banksPerChannel;
  addr.bank = flatBank % geometry.banksPerChannel;
  // Ranks are part of the coordinate but nothing distributes over them yet.
  addr.rank = 0;
  addr.subarray = 0;
  addr.row = 0;
  return addr;
}

RowAddress AddressAllocator::allocate(int64_t flatBank, int64_t numRows) {
  assert(numRows > 0 && "an allocation covers at least one row");
  RowAddress addr = bankAddress(flatBank);
  int64_t &next = nextRow[flatBank];
  addr.row = next;
  next += numRows;
  return addr;
}

int64_t AddressAllocator::rowsAllocated(int64_t flatBank) const {
  auto it = nextRow.find(flatBank);
  return it == nextRow.end() ? 0 : it->second;
}
