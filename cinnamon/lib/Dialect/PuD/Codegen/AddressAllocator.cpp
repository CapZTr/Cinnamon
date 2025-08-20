#include "cinm-mlir/Dialect/PuD/Codegen/AddressAllocator.h"


RowAddress AddressAllocator::allocate(int64_t numRows) {
  assert(numRows <= MAX_ROW);

  auto subArrayID = checkSpace(numRows);
  auto saLocalID = subArrayID % 64;

  int64_t channel = currentChannel;
  int64_t rank = currentRank;
  int64_t bank = currentBank;
  int64_t sa = currentSubarray;
  int64_t row = currentRow;
  if (saLocalID == currentBank) {
    currentRow += numRows;
    row = currentRow;
  } else if (subArrayID == -1) {
    currentRow = 0;
    ++currentSubarray;
    if (currentSubarray > MAX_SUBARRAY) {
      currentSubarray = 0;
      ++currentBank;
      if (currentBank > MAX_BANK) {
        currentBank = 0;
        ++currentRank;
        if (currentRank > MAX_RANK) {
          currentRank = 0;
          ++currentChannel;
          if (currentChannel > MAX_CHANNEL) {
          }
        }
      }
    }
    channel = currentChannel;
    rank = currentRank;
    bank = currentRank;
    sa = currentSubarray;
    row = currentRow;
  } else {
    int64_t channelSANum = 2 * 8 * 64;
    channel = subArrayID < channelSANum ? 0 : 1;

    auto saChannelID = subArrayID % channelSANum;
    int64_t rankSANum = 8 * 64;

    rank = saChannelID < rankSANum ? 0 : 1;
    auto saRankID = saChannelID % rankSANum;

    bank = saRankID / 8;

    sa = saLocalID;
    auto space = restSpace.lookup(subArrayID);
    row = MAX_ROW - space;
  }

  RowAddress addr = { 
      channel, 
      rank, 
      bank, 
      sa, 
      row-numRows
  };

  int64_t saToUpdate = channel * 2 + rank * 2 + bank * 8 + sa;
  int64_t rest = MAX_ROW - row - numRows;
  restSpace[saToUpdate] = rest;

  return addr;
}

RowAddress AddressAllocator::getRowFromOffset(const RowAddress &base, const int64_t offset) {
  auto rowID = base.row + offset;
  assert(rowID <= MAX_ROW);
  return {base.channel, base.rank, base.bank, base.subarray, rowID};
}

int AddressAllocator::checkSpace(int64_t numRows) const {
  if (restSpace.empty()) {
    return 0;
  }
  int subArrayID = -1;
  for (const auto &pair : restSpace) {
    if (numRows <= pair.second)
      subArrayID = pair.first;
  }
  return subArrayID;
}
