#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace mlir::pud {

/// Where a row lives, in the terms the memory is organised in.
///
/// All five coordinates are carried whether or not the compiler currently
/// reasons about them: the mapping decides which bank a computation runs in and
/// nothing below it distributes rows across subarrays yet, so `subarray` stays
/// zero, but a row is still identified by the full coordinate and modelling the
/// remaining dimensions later does not change what an address is.
struct RowAddress {
  int64_t channel = 0;
  int64_t rank = 0;
  int64_t bank = 0;
  int64_t subarray = 0;
  int64_t row = 0;

  bool inSameSubArray(const RowAddress &other) const {
    return channel == other.channel && rank == other.rank &&
           bank == other.bank && subarray == other.subarray;
  }

  std::string str() const {
    return std::to_string(channel) + " " + std::to_string(rank) + " " +
           std::to_string(bank) + " " + std::to_string(subarray) + " " +
           std::to_string(row);
  }
};

/// How the memory is organised, as the mapping understood it.
///
/// The mapping chooses banks from a flat range and prices data movement by
/// whether a transfer crosses a channel, so it already knows this; carrying the
/// same numbers here is what makes a bank index mean the same thing on both
/// sides of the lowering.
struct MemoryGeometry {
  /// Banks addressable in total, across every channel.
  int64_t totalBanks = 32;
  /// Banks in one channel. A flat bank index splits into a channel and a bank
  /// within it on this boundary.
  int64_t banksPerChannel = 32;
  /// Ranks per channel. Not yet distributed over -- every address takes rank 0
  /// -- but part of the coordinate.
  int64_t ranksPerChannel = 1;
};

/// Hands out rows for a computation, and resolves the flat bank indices the
/// mapping works in into coordinates.
///
/// Rows are handed out in sequence within one bank. Nothing here spreads them
/// across subarrays: a row op reads and writes rows of a single subarray, and
/// choosing between subarrays only becomes a decision once the mapping reasons
/// about them.
class AddressAllocator {
public:
  AddressAllocator() = default;
  explicit AddressAllocator(MemoryGeometry geometry) : geometry(geometry) {}

  /// Coordinate of a flat bank index, as the mapping numbers banks.
  RowAddress bankAddress(int64_t flatBank) const;

  /// Reserves `numRows` consecutive rows in `flatBank` and returns the first.
  RowAddress allocate(int64_t flatBank, int64_t numRows);

  /// The row `offset` rows after `base`, in the same bank.
  RowAddress getRowFromOffset(const RowAddress &base, int64_t offset) const {
    RowAddress addr = base;
    addr.row += offset;
    return addr;
  }

  /// First row not yet handed out in `flatBank`.
  int64_t rowsAllocated(int64_t flatBank) const;

  void reset() { nextRow.clear(); }

  const MemoryGeometry &getGeometry() const { return geometry; }

private:
  MemoryGeometry geometry;
  /// Rows handed out per flat bank index. A plain map rather than a single
  /// counter: banks are allocated independently, and two banks running the same
  /// computation should get the same rows.
  std::map<int64_t, int64_t> nextRow;
};

} // namespace mlir::pud
