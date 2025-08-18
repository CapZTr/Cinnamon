#pragma once

#include <llvm/ADT/DenseMap.h>

#include <cassert>
#include <cstdint>
#include <format>
#include <string>


struct RowAddress {
  int64_t channel;
  int64_t rank;
  int64_t bank;
  int64_t subarray;
  int64_t row;

  bool inSameSubArray(const RowAddress &other) {
    return this->channel == other.channel
        && this->rank == other.rank
        && this->bank == other.bank
        && this->subarray == other.subarray;
  }

  std::string str() {
    return std::format("{} {} {} {} {}",
        channel,
        rank,
        bank,
        subarray,
        row);
  }
};

class AddressAllocator {
public:
  AddressAllocator() = default;

  RowAddress allocate(int64_t numRows);

  RowAddress getRowFromOffset(const RowAddress &base, const int64_t offset);

  void reset() {
    currentChannel = 0;
    currentRank = 0;
    currentBank = 0;
    currentSubarray = 0;
    currentRow = 0;
  }

  int64_t getMaxColumnNum() const {
    return MAX_COLUMN;
  }

private:
  const int64_t MAX_CHANNEL = 1;
  const int64_t MAX_RANK = 1;
  const int64_t MAX_BANK = 7;
  const int64_t MAX_SUBARRAY = 63;
  const int64_t MAX_ROW = 1005;
  const int64_t MAX_COLUMN = 8192;

  int64_t currentChannel = 0;
  int64_t currentRank = 0;
  int64_t currentBank = 0;
  int64_t currentSubarray = 0;
  int64_t currentRow = 0;

  llvm::DenseMap<int64_t, int64_t> restSpace;

  int checkSpace(int64_t numRows) const;

};
