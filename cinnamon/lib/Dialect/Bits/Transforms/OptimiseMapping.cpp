#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"
#include "gurobi_c++.h"
#include "gurobi_c.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <vector>

namespace mlir::bits {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSOPTMISEMAPPINGPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

constexpr int64_t kNumBanks = 32;
constexpr int64_t kCloneCostPerBit = 14;
constexpr bool kEnableVerboseMappingLog = true;

struct Node {
  Operation *op = nullptr;
  int64_t duration = 0;
  bool isContainer = false;
  int parentForNode = -1;
  int64_t loopDepth = 0;
  llvm::SmallVector<int, 2> preds;
  llvm::SmallVector<int, 4> succs;
};

struct DAG {
  llvm::SmallVector<Node> nodes;
  llvm::DenseMap<Operation *, int> opToNodeID;
};

struct MappingSolution {
  std::vector<int> chosenBank;
  std::vector<int64_t> startTime;
  std::vector<int64_t> endTime;
};

struct TransferEvent {
  int srcNodeId = -1;
  int dstNodeId = -1;
  int srcBank = -1;
  int dstBank = -1;
  int64_t start = 0;
  int64_t end = 0;
  int64_t valueId = -1;
};

struct MuliEvent {
  int64_t lhsBitwidth = 0;
  int64_t rhsBitwidth = 0;
  int64_t start = 0;
  int64_t end = 0;
  SmallVector<int64_t> banks;
};

struct AddiEvent {
  int64_t lhsBitwidth = 0;
  int64_t rhsBitwidth = 0;
  int64_t start = 0;
  int64_t end = 0;
  SmallVector<int64_t> banks;
};

struct RowCopyEvent {
  SmallVector<std::pair<int64_t, int64_t>> edges;
  int64_t bitwidth = 0;
  int64_t start = 0;
  int64_t end = 0;
};

struct FlattenedEvent {
  enum class Kind { Muli, Addi, RowCopy };
  Kind kind = Kind::Muli;
  int64_t start = 0;
  int64_t end = 0;
  MuliEvent muli;
  AddiEvent addi;
  RowCopyEvent rowCopy;
};

// void modelAndSolveILP(const DAG &dag) {
static bool isSliceType(Type ty) { return isa<SliceType>(ty); }

static bool hasOnlySliceOperandsAndResults(Operation *op) {
  for (Value operand : op->getOperands()) {
    if (!isSliceType(operand.getType()))
      return false;
  }
  if (op->getNumResults() == 0)
    return false;
  for (Value result : op->getResults()) {
    if (!isSliceType(result.getType()))
      return false;
  }
  return true;
}

static bool isSupportedBitsSliceComputeOp(Operation *op) {
  if (op->getName().getDialectNamespace() != "bits")
    return false;

  if (!hasOnlySliceOperandsAndResults(op))
    return false;

  if (op->getNumOperands() != 1 && op->getNumOperands() != 2)
    return false;

  static const llvm::StringSet<> kExcludedOps = {
      "bits.transpose",  "bits.assemble",      "bits.extract_slice",
      "bits.extract_row","bits.insert_row",    "bits.insert_slice",
      "bits.extract_and_replicate",            "bits.ext_i",
      "bits.split_vertically",                 "bits.merge_vertically",
      "bits.create_slice", "bits.shift_up",    "bits.matmul",
      "bits.matvecmul"};
  return !kExcludedOps.contains(op->getName().getStringRef());
}

static int64_t getSliceBitwidth(Value value) {
  auto sliceTy = dyn_cast<SliceType>(value.getType());
  return sliceTy ? sliceTy.getBitWidth() : 0;
}

static int64_t getComputeLatency(Operation *op) {
  if (auto add = dyn_cast<AddIOp>(op))
    return 8 * getSliceBitwidth(add.getResult());
  if (auto mul = dyn_cast<MulIOp>(op))
    return 12 * getSliceBitwidth(mul.getLhs()) * getSliceBitwidth(mul.getRhs()) -
           12 * getSliceBitwidth(mul.getLhs()) + 4;
  if (auto andOp = dyn_cast<AndOp>(op))
    return 4 * getSliceBitwidth(andOp.getResult());
  if (auto orOp = dyn_cast<OrOp>(op))
    return 4 * getSliceBitwidth(orOp.getResult());
  // if (auto xorOp = dyn_cast<XOrOp>(op))
  //   return 3 * getSliceBitwidth(xorOp.getResult());

  if (op->getNumResults() == 0)
    return 0;
  return 6 * getSliceBitwidth(op->getResult(0));
}

static FailureOr<int64_t> getConstTripCount(scf::ForOp forOp);
static SmallVector<int64_t> buildBankGroup(int64_t baseBank, int64_t numBanks);
static bool isSliceReductionFor(scf::ForOp forOp);
static int64_t estimatePeakBanksPerIteration(scf::ForOp forOp);
static int64_t estimateBestParallelBanksForLoop(scf::ForOp forOp);

static bool isMuliOrAddi(Operation *op) {
  return isa<MulIOp>(op) || isa<AddIOp>(op);
}

static SmallVector<int64_t> getReduceBanksFromAttr(scf::ForOp forOp) {
  SmallVector<int64_t> banks;
  auto treeAttr = forOp->getAttrOfType<ArrayAttr>("bits.reduce_tree");
  if (!treeAttr || treeAttr.empty())
    return banks;
  auto firstLevel = dyn_cast<ArrayAttr>(treeAttr[0]);
  if (!firstLevel)
    return banks;
  llvm::DenseSet<int64_t> uniqueBanks;
  for (Attribute pairAttr : firstLevel) {
    auto pair = dyn_cast<DictionaryAttr>(pairAttr);
    if (!pair)
      continue;
    auto src = dyn_cast_or_null<IntegerAttr>(pair.get("src_bank"));
    auto dst = dyn_cast_or_null<IntegerAttr>(pair.get("dst_bank"));
    if (src)
      uniqueBanks.insert(src.getInt());
    if (dst)
      uniqueBanks.insert(dst.getInt());
  }
  for (int64_t bank : uniqueBanks)
    banks.push_back(bank);
  llvm::sort(banks);
  return banks;
}

// Binary trace file layout (shared with C reader).
// Header: 32 bytes
struct TraceHeader {
  char     magic[8];      // "CIMTRACE"
  uint32_t version;       // = 1
  uint32_t record_size;   // = sizeof(TraceRecord) = 24
  uint64_t num_records;
  int64_t  last_end_time;
};
static_assert(sizeof(TraceHeader) == 32, "TraceHeader size mismatch");
 
// Record: 32 bytes, one entry per merged event (MULI/ADDI) or per edge (ROWCOPY).
// MULI/ADDI banks are encoded as a bitmask in `banks` (kNumBanks==32 fits uint32_t).
// ROWCOPY emits one record per (src,dst) edge; `banks` is unused (0).
struct TraceRecord {
  int64_t  start;      // offset  0, 8B: cycle start
  int64_t  end;        // offset  8, 8B: cycle end
  uint32_t banks;      // offset 16, 4B: MULI/ADDI bank bitmask; 0 for ROWCOPY
  uint16_t lhs_bw;     // offset 20, 2B: MULI/ADDI→lhsBitwidth, ROWCOPY→bitwidth
  uint16_t rhs_bw;     // offset 22, 2B: MULI/ADDI→rhsBitwidth, ROWCOPY→0
  uint8_t  kind;       // offset 24, 1B: 0=MULI, 1=ADDI, 2=ROWCOPY
  uint8_t  src;        // offset 25, 1B: ROWCOPY→srcBank, others→0
  uint8_t  dst;        // offset 26, 1B: ROWCOPY→dstBank, others→0
  uint8_t  pad[5];     // offset 27, 5B: explicit pad to 32 bytes
};
static_assert(sizeof(TraceRecord) == 32, "TraceRecord size mismatch");

static void dumpFlattenedEventSequence(func::FuncOp func) {
  std::map<int64_t, std::vector<FlattenedEvent>> eventsByStartTime;
  std::map<int64_t, std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>, size_t>>
      eventIndexByStartTime;

  auto recordEvent = [&](const FlattenedEvent &evt) {
    auto &events = eventsByStartTime[evt.start];
    auto &eventIndex = eventIndexByStartTime[evt.start];

    if (evt.kind == FlattenedEvent::Kind::Muli) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.muli.lhsBitwidth, evt.muli.rhsBitwidth,
                                 evt.muli.start, evt.muli.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.muli.banks.append(evt.muli.banks.begin(), evt.muli.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::Addi) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.addi.lhsBitwidth, evt.addi.rhsBitwidth,
                                 evt.addi.start, evt.addi.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.addi.banks.append(evt.addi.banks.begin(), evt.addi.banks.end());
      }
      return;
    }

    auto key = std::make_tuple(static_cast<int64_t>(evt.kind), evt.rowCopy.bitwidth,
                               int64_t{0}, evt.rowCopy.start, evt.rowCopy.end);
    auto [it, inserted] = eventIndex.try_emplace(key, events.size());
    if (inserted) {
      events.push_back(evt);
    } else {
      auto &existing = events[it->second];
      existing.rowCopy.edges.append(evt.rowCopy.edges.begin(), evt.rowCopy.edges.end());
    }
  };

  auto getAncestorWaveRepeat = [&](scf::ForOp forOp) {
    int64_t repeat = 1;
    for (Operation *parent = forOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto parentFor = dyn_cast<scf::ForOp>(parent);
      if (!parentFor)
        continue;
      auto tripMaybe = getConstTripCount(parentFor);
      if (failed(tripMaybe))
        continue;
      int64_t tripCount = *tripMaybe;
      int64_t banksPerIter = 1;
      if (isSliceReductionFor(parentFor)) {
        banksPerIter =
            std::max<int64_t>(1, estimateBestParallelBanksForLoop(parentFor));
      } else {
        banksPerIter =
            std::max<int64_t>(1, estimatePeakBanksPerIteration(parentFor));
      }
      int64_t concurrentIters = std::max<int64_t>(1, kNumBanks / banksPerIter);
      int64_t waves = std::max<int64_t>(1, (tripCount + concurrentIters - 1) / concurrentIters);
      if (repeat > std::numeric_limits<int64_t>::max() / waves)
        return std::numeric_limits<int64_t>::max();
      repeat *= waves;
    }
    return repeat;
  };

  auto getNearestAncestorConcurrentGroups = [&](scf::ForOp forOp,
                                                int64_t banksPerGroup) {
    for (Operation *parent = forOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      auto parentFor = dyn_cast<scf::ForOp>(parent);
      if (!parentFor)
        continue;
      auto tripMaybe = getConstTripCount(parentFor);
      if (failed(tripMaybe))
        return int64_t{1};
      int64_t tripCount = *tripMaybe;
      int64_t parentParallel = tripCount;
      if (auto pf =
              parentFor->getAttrOfType<IntegerAttr>("bits.parallel_factor")) {
        parentParallel = std::max<int64_t>(1, pf.getInt());
      }
      int64_t bankLimited =
          std::max<int64_t>(1, kNumBanks / std::max<int64_t>(1, banksPerGroup));
      return std::max<int64_t>(
          1, std::min<int64_t>({tripCount, parentParallel, bankLimited}));
    }
    return int64_t{1};
  };

  func.walk([&](scf::ForOp forOp) {
    auto tripMaybe = getConstTripCount(forOp);
    if (failed(tripMaybe))
      return;
    int64_t tripCount = *tripMaybe;
    if (tripCount <= 0)
      return;

    int64_t parallelFactor = 1;
    if (auto pf = forOp->getAttrOfType<IntegerAttr>("bits.parallel_factor"))
      parallelFactor = std::max<int64_t>(1, pf.getInt());

    int64_t baseBank = 0;
    if (auto b = forOp->getAttrOfType<IntegerAttr>("bits.bank_id"))
      baseBank = b.getInt();

    SmallVector<int64_t> activeBanks;
    auto banksFromTree = getReduceBanksFromAttr(forOp);
    if (!banksFromTree.empty()) {
      activeBanks = banksFromTree;
    } else {
      activeBanks = buildBankGroup(baseBank, parallelFactor);
    }
    int64_t effectiveParallel =
        std::max<int64_t>(1, static_cast<int64_t>(activeBanks.size()));
    int64_t itersPerBank = std::max<int64_t>(
        1, (tripCount + effectiveParallel - 1) / effectiveParallel);

    Operation *reduceProducer = nullptr;
    if (auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
      if (yield.getNumOperands() == 1)
        reduceProducer = yield.getOperand(0).getDefiningOp();
    }

    SmallVector<Operation *> linearOps;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isMuliOrAddi(&op))
        continue;
      linearOps.push_back(&op);
    }
    if (linearOps.empty())
      return;

    DenseMap<int64_t, int64_t> bankReady;
    for (int64_t bank : activeBanks)
      bankReady[bank] = 0;
    std::vector<FlattenedEvent> localEvents;

    int64_t reduceBitwidth = 0;
    int64_t reduceLatency = 0;
    if (reduceProducer && isMuliOrAddi(reduceProducer)) {
      reduceLatency = getComputeLatency(reduceProducer);
      if (reduceProducer->getNumOperands() >= 1)
        reduceBitwidth = getSliceBitwidth(reduceProducer->getOperand(0));
    }
    int64_t rowCopyLatency = kCloneCostPerBit * std::max<int64_t>(1, reduceBitwidth);

    for (int64_t bank : activeBanks) {
      int64_t cursor = bankReady[bank];
      for (int64_t iter = 0; iter < itersPerBank; ++iter) {
        for (Operation *op : linearOps) {
          if (op == reduceProducer)
            continue;
          int64_t lat = getComputeLatency(op);
          int64_t start = cursor;
          int64_t end = start + lat;
          if (auto mul = dyn_cast<MulIOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Muli;
            evt.start = start;
            evt.end = end;
            evt.muli.lhsBitwidth = getSliceBitwidth(mul.getLhs());
            evt.muli.rhsBitwidth = getSliceBitwidth(mul.getRhs());
            evt.muli.start = start;
            evt.muli.end = end;
            evt.muli.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (auto add = dyn_cast<AddIOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Addi;
            evt.start = start;
            evt.end = end;
            evt.addi.lhsBitwidth = getSliceBitwidth(add.getLhs());
            evt.addi.rhsBitwidth = getSliceBitwidth(add.getRhs());
            evt.addi.start = start;
            evt.addi.end = end;
            evt.addi.banks.push_back(bank);
            localEvents.push_back(evt);
          }
          cursor = end;
        }
        if (reduceProducer && iter > 0) {
          int64_t start = cursor;
          int64_t end = start + reduceLatency;
          if (auto add = dyn_cast<AddIOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Addi;
            evt.start = start;
            evt.end = end;
            evt.addi.lhsBitwidth = getSliceBitwidth(add.getLhs());
            evt.addi.rhsBitwidth = getSliceBitwidth(add.getRhs());
            evt.addi.start = start;
            evt.addi.end = end;
            evt.addi.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (auto mul = dyn_cast<MulIOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Muli;
            evt.start = start;
            evt.end = end;
            evt.muli.lhsBitwidth = getSliceBitwidth(mul.getLhs());
            evt.muli.rhsBitwidth = getSliceBitwidth(mul.getRhs());
            evt.muli.start = start;
            evt.muli.end = end;
            evt.muli.banks.push_back(bank);
            localEvents.push_back(evt);
          }
          cursor = end;
        }
      }
      bankReady[bank] = cursor;
    }

    auto treeAttr = forOp->getAttrOfType<ArrayAttr>("bits.reduce_tree");
    if (treeAttr && reduceProducer) {
      for (Attribute levelAttr : treeAttr) {
        auto level = dyn_cast<ArrayAttr>(levelAttr);
        if (!level)
          continue;
        for (Attribute pairAttr : level) {
          auto pair = dyn_cast<DictionaryAttr>(pairAttr);
          if (!pair)
            continue;
          auto srcAttr = dyn_cast_or_null<IntegerAttr>(pair.get("src_bank"));
          auto dstAttr = dyn_cast_or_null<IntegerAttr>(pair.get("dst_bank"));
          if (!srcAttr || !dstAttr)
            continue;
          int64_t srcBank = srcAttr.getInt();
          int64_t dstBank = dstAttr.getInt();
          int64_t copyStart = bankReady[srcBank];
          int64_t copyEnd = copyStart + rowCopyLatency;
          FlattenedEvent copyEvt;
          copyEvt.kind = FlattenedEvent::Kind::RowCopy;
          copyEvt.start = copyStart;
          copyEvt.end = copyEnd;
          copyEvt.rowCopy.bitwidth = reduceBitwidth;
          copyEvt.rowCopy.start = copyStart;
          copyEvt.rowCopy.end = copyEnd;
          copyEvt.rowCopy.edges.push_back({srcBank, dstBank});
          localEvents.push_back(copyEvt);
          int64_t addStart = std::max<int64_t>(bankReady[dstBank], copyEnd);
          int64_t addEnd = addStart + reduceLatency;
          if (auto add = dyn_cast<AddIOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Addi;
            evt.start = addStart;
            evt.end = addEnd;
            evt.addi.lhsBitwidth = getSliceBitwidth(add.getLhs());
            evt.addi.rhsBitwidth = getSliceBitwidth(add.getRhs());
            evt.addi.start = addStart;
            evt.addi.end = addEnd;
            evt.addi.banks.push_back(dstBank);
            localEvents.push_back(evt);
          } else if (auto mul = dyn_cast<MulIOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Muli;
            evt.start = addStart;
            evt.end = addEnd;
            evt.muli.lhsBitwidth = getSliceBitwidth(mul.getLhs());
            evt.muli.rhsBitwidth = getSliceBitwidth(mul.getRhs());
            evt.muli.start = addStart;
            evt.muli.end = addEnd;
            evt.muli.banks.push_back(dstBank);
            localEvents.push_back(evt);
          }
          bankReady[dstBank] = addEnd;
        }
      }
    }

    int64_t localLatency = 0;
    for (const FlattenedEvent &evt : localEvents)
      localLatency = std::max<int64_t>(localLatency, evt.end);
    int64_t bankGroupStride =
        std::max<int64_t>(1, estimateBestParallelBanksForLoop(forOp));
    int64_t groupsPerWave =
        getNearestAncestorConcurrentGroups(forOp, bankGroupStride);
    int64_t repeatCount = std::max<int64_t>(1, getAncestorWaveRepeat(forOp));
    for (int64_t wave = 0; wave < repeatCount; ++wave) {
      int64_t offset = wave * localLatency;
      for (int64_t group = 0; group < groupsPerWave; ++group) {
        int64_t bankOffset = group * bankGroupStride;
        for (const FlattenedEvent &evt : localEvents) {
          FlattenedEvent shifted = evt;
          shifted.start += offset;
          shifted.end += offset;
          if (shifted.kind == FlattenedEvent::Kind::Muli) {
            shifted.muli.start = shifted.start;
            shifted.muli.end = shifted.end;
            shifted.muli.banks[0] = (shifted.muli.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::Addi) {
            shifted.addi.start = shifted.start;
            shifted.addi.end = shifted.end;
            shifted.addi.banks[0] = (shifted.addi.banks[0] + bankOffset) % kNumBanks;
          } else {
            shifted.rowCopy.start = shifted.start;
            shifted.rowCopy.end = shifted.end;
            shifted.rowCopy.edges[0].first =
                (shifted.rowCopy.edges[0].first + bankOffset) % kNumBanks;
            shifted.rowCopy.edges[0].second =
                (shifted.rowCopy.edges[0].second + bankOffset) % kNumBanks;
          }
          recordEvent(shifted);
        }
      }
    }
  });

  // llvm::outs() << "=== flattened-op-sequence ===\n";
  // for (const FlattenedEvent &event : events) {
  //   switch (event.kind) {
  //   case FlattenedEvent::Kind::Muli:
  //     llvm::outs() << "muli(lhs_bw=" << event.muli.lhsBitwidth
  //                  << ", rhs_bw=" << event.muli.rhsBitwidth
  //                  << ", start=" << event.muli.start
  //                  << ", end=" << event.muli.end
  //                  << ", bank=" << event.muli.bankId << ")\n";
  //     break;
  //   case FlattenedEvent::Kind::Addi:
  //     llvm::outs() << "addi(lhs_bw=" << event.addi.lhsBitwidth
  //                  << ", rhs_bw=" << event.addi.rhsBitwidth
  //                  << ", start=" << event.addi.start
  //                  << ", end=" << event.addi.end
  //                  << ", bank=" << event.addi.bankId << ")\n";
  //     break;
  //   case FlattenedEvent::Kind::RowCopy:
  //     llvm::outs() << "row_copy(src_bank=" << event.rowCopy.srcBankId
  //                  << ", dst_bank=" << event.rowCopy.dstBankId
  //                  << ", bitwidth=" << event.rowCopy.bitwidth
  //                  << ", start=" << event.rowCopy.start
  //                  << ", end=" << event.rowCopy.end << ")\n";
  //     break;
  //   }
  // }

  constexpr size_t kTraceBufCap = 1u << 16; // 64 K records per fwrite
  std::vector<TraceRecord> traceBuf(kTraceBufCap);
  size_t   traceBufLen  = 0;
  uint64_t totalRecords = 0;
  int64_t  lastEndTime  = 0;
 
  const char *tracePath =
      "/home/tianruiz/cimdram/traces/trace.bin";
  FILE *traceFp = std::fopen(tracePath, "wb");
  if (!traceFp) {
    llvm::errs() << "error: cannot open trace file: " << tracePath << "\n";
    return;
  }
 
  // Write placeholder header; num_records / last_end_time filled in at end.
  TraceHeader hdr{};
  std::memcpy(hdr.magic, "CIMTRACE", 8);
  hdr.version     = 1;
  hdr.record_size = static_cast<uint32_t>(sizeof(TraceRecord));
  std::fwrite(&hdr, sizeof(TraceHeader), 1, traceFp);
 
  auto traceFlush = [&]() {
    std::fwrite(traceBuf.data(), sizeof(TraceRecord), traceBufLen, traceFp);
    totalRecords += traceBufLen;
    traceBufLen = 0;
  };
  auto traceEmit = [&](TraceRecord r) {
    traceBuf[traceBufLen++] = r;
    if (traceBufLen == kTraceBufCap)
      traceFlush();
  };

  for (const auto &[startTime, sameStartEvents] : eventsByStartTime) {
    (void)startTime;
    for (const FlattenedEvent &event : sameStartEvents) {
      TraceRecord r{};
      r.start = event.start;
      r.end   = event.end;
      switch (event.kind) {
      case FlattenedEvent::Kind::Muli:
        r.kind   = 0;
        r.lhs_bw = static_cast<uint16_t>(event.muli.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.muli.rhsBitwidth);
        for (int64_t bank : event.muli.banks)
          r.banks |= (1u << bank);  // fold all banks into bitmask, one record
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Addi:
        r.kind   = 1;
        r.lhs_bw = static_cast<uint16_t>(event.addi.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.addi.rhsBitwidth);
        for (int64_t bank : event.addi.banks)
          r.banks |= (1u << bank);  // fold all banks into bitmask, one record
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::RowCopy:
        r.kind   = 2;
        r.lhs_bw = static_cast<uint16_t>(event.rowCopy.bitwidth);
        for (const auto &[src, dst] : event.rowCopy.edges) {
          r.src = static_cast<uint8_t>(src);
          r.dst = static_cast<uint8_t>(dst);
          traceEmit(r);  // one record per edge (edges carry pairing info)
        }
        break;
      }
      lastEndTime = event.end;
    }
  }
  traceFlush();
 
  // Back-fill header with final counts.
  hdr.num_records   = totalRecords;
  hdr.last_end_time = lastEndTime;
  std::rewind(traceFp);
  std::fwrite(&hdr, sizeof(TraceHeader), 1, traceFp);
  std::fclose(traceFp);

  llvm::outs() << "last_end_time=" << lastEndTime << "\n";
}

static FailureOr<int64_t> getConstTripCount(scf::ForOp forOp) {
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step)
    return failure();
  int64_t lbv = lb.value();
  int64_t ubv = ub.value();
  int64_t stepv = step.value();
  if (stepv <= 0 || ubv < lbv)
    return failure();
  int64_t diff = ubv - lbv;
  if (diff % stepv != 0)
    return failure();
  return diff / stepv;
}

static bool isSliceReductionFor(scf::ForOp forOp) {
  if (forOp.getNumRegionIterArgs() != 1)
    return false;
  return isSliceType(forOp.getRegionIterArgs().front().getType());
}

static bool hasNestedForOp(scf::ForOp forOp) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<scf::ForOp>(op))
      return true;
  }
  return false;
}

static int64_t estimateForLatency(scf::ForOp forOp);

static int64_t estimateLoopBodyLatency(scf::ForOp forOp) {
  int64_t bodyLatency = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
      bodyLatency += estimateForLatency(innerFor);
      continue;
    }
    if (isSupportedBitsSliceComputeOp(&op))
      bodyLatency += getComputeLatency(&op);
  }
  return bodyLatency;
}

static int64_t estimateForLatency(scf::ForOp forOp) {
  FailureOr<int64_t> maybeTripCount = getConstTripCount(forOp);
  if (failed(maybeTripCount))
    return 0;
  int64_t tripCount = *maybeTripCount;
  if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tripCount)))
    return 0;

  int64_t bodyLatency = estimateLoopBodyLatency(forOp);
  if (bodyLatency == 0)
    return 0;

  if (!isSliceReductionFor(forOp)) {
    // Non-reduction loops can run multiple iterations in parallel, but each
    // iteration may consume more than one bank when the body contains nested
    // parallel/reduction loops. Bound throughput by the per-iteration bank
    // footprint inferred from nested loop planning.
    int64_t banksPerIter = std::max<int64_t>(1, estimatePeakBanksPerIteration(forOp));
    int64_t concurrentIters = std::max<int64_t>(1, kNumBanks / banksPerIter);
    return bodyLatency * ((tripCount + concurrentIters - 1) / concurrentIters);
  }

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return bodyLatency * tripCount;

  Operation *reduceProducer = yield.getOperand(0).getDefiningOp();
  int64_t reduceLatency =
      (reduceProducer && isSupportedBitsSliceComputeOp(reduceProducer))
          ? getComputeLatency(reduceProducer)
          : 0;
  int64_t reduceBitwidth = getSliceBitwidth(yield.getOperand(0));
  int64_t cloneLatency = kCloneCostPerBit * reduceBitwidth;

  int64_t best = std::numeric_limits<int64_t>::max();
  for (int64_t banks = 1; banks <= std::min<int64_t>(tripCount, kNumBanks);
       banks <<= 1) {
    int64_t iterPerBank = (tripCount + banks - 1) / banks;
    // Each bank: n iterations of pre-reduce work (body minus the reduce op),
    // then (n-1) intra-bank reductions to fold n partial results into one.
    // Only after that does inter-bank tree reduction begin.
    int64_t preReduceLatency = bodyLatency - reduceLatency;
    int64_t local = iterPerBank * preReduceLatency
                  + std::max<int64_t>(0, iterPerBank - 1) * reduceLatency;
    int64_t reduceTreeLevels = llvm::Log2_64(banks);
    int64_t reduceCost = reduceTreeLevels * (cloneLatency + reduceLatency);
    best = std::min(best, local + reduceCost);
  }
  return best == std::numeric_limits<int64_t>::max() ? 0 : best;
}

static int64_t estimatePeakBanksPerIteration(scf::ForOp forOp) {
  int64_t peakBanks = 1;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
      peakBanks = std::max<int64_t>(
          peakBanks, std::max<int64_t>(1, estimateBestParallelBanksForLoop(innerFor)));
    }
  }
  return peakBanks;
}

static int64_t estimateBestParallelBanksForLoop(scf::ForOp forOp) {
  FailureOr<int64_t> maybeTripCount = getConstTripCount(forOp);
  if (failed(maybeTripCount))
    return 1;
  int64_t tripCount = *maybeTripCount;
  if (!llvm::isPowerOf2_64(static_cast<uint64_t>(tripCount)))
    return 1;

  int64_t bodyLatency = estimateLoopBodyLatency(forOp);
  if (bodyLatency <= 0)
    return 1;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  int64_t reduceBitwidth =
      (yield && yield.getNumOperands() == 1) ? getSliceBitwidth(yield.getOperand(0))
                                             : 0;
  Operation *reduceProducer =
      (yield && yield.getNumOperands() == 1) ? yield.getOperand(0).getDefiningOp()
                                             : nullptr;
  int64_t reduceLatency =
      (reduceProducer && isSupportedBitsSliceComputeOp(reduceProducer))
          ? getComputeLatency(reduceProducer)
          : 0;
  int64_t cloneLatency = kCloneCostPerBit * reduceBitwidth;

  int64_t bestBanks = 1;
  int64_t bestCost = std::numeric_limits<int64_t>::max();
  for (int64_t banks = 1; banks <= std::min<int64_t>(tripCount, kNumBanks);
       banks <<= 1) {
    int64_t iterPerBank = (tripCount + banks - 1) / banks;
    int64_t local;
    if (isSliceReductionFor(forOp) && reduceLatency > 0) {
      // Same model as estimateForLatency: n pre-reduce iters + (n-1)
      // intra-bank reductions.  The reduce op is NOT counted in every iteration.
      int64_t preReduceLatency = bodyLatency - reduceLatency;
      local = iterPerBank * preReduceLatency
            + std::max<int64_t>(0, iterPerBank - 1) * reduceLatency;
    } else {
      local = iterPerBank * bodyLatency;
    }
    int64_t reduceTreeLevels = llvm::Log2_64(banks);
    int64_t reduceCost =
        isSliceReductionFor(forOp) ? reduceTreeLevels * (cloneLatency + reduceLatency)
                                   : 0;
    int64_t totalCost = local + reduceCost;
    if (totalCost < bestCost) {
      bestCost = totalCost;
      bestBanks = banks;
    }
  }
  return bestBanks;
}

static SmallVector<int64_t> buildBankGroup(int64_t baseBank, int64_t numBanks) {
  SmallVector<int64_t> banks;
  if (numBanks <= 0)
    return banks;
  banks.reserve(numBanks);
  for (int64_t i = 0; i < numBanks; ++i)
    banks.push_back((baseBank + i) % kNumBanks);
  return banks;
}

static DenseMap<int64_t, int64_t>
buildTreeReduceCountPerBank(int64_t baseBank, int64_t numBanks) {
  DenseMap<int64_t, int64_t> reduceCounts;
  if (numBanks <= 1)
    return reduceCounts;

  SmallVector<int64_t> active = buildBankGroup(baseBank, numBanks);
  while (active.size() > 1) {
    SmallVector<int64_t> next;
    next.reserve((active.size() + 1) / 2);
    for (size_t i = 0; i < active.size(); i += 2) {
      int64_t left = active[i];
      if (i + 1 < active.size()) {
        reduceCounts[left] += 1;
      }
      next.push_back(left);
    }
    active = std::move(next);
  }
  return reduceCounts;
}

struct RegionPlan {
  int64_t regionCount = 1;
  int64_t banksPerRegion = 1;
  int64_t iterPerRegion = 1;
  int64_t unrollFactor = 1;
  int64_t computeOpsPerRegion = 0;
  int64_t estimatedMakespan = 0;
  SmallVector<SmallVector<int64_t>> regionBanks;
};

static SmallVector<int64_t> getPowerOfTwoChoices(int64_t upper) {
  SmallVector<int64_t> choices;
  for (int64_t v = 1; v <= upper; v <<= 1)
    choices.push_back(v);
  return choices;
}

static RegionPlan planLeafForRegions(scf::ForOp forOp, int64_t baseBank,
                                     int64_t availBanks) {
  RegionPlan best;
  best.estimatedMakespan = std::numeric_limits<int64_t>::max();

  FailureOr<int64_t> maybeTrip = getConstTripCount(forOp);
  if (failed(maybeTrip) || *maybeTrip <= 0 || availBanks <= 0) {
    best.estimatedMakespan = 0;
    return best;
  }
  int64_t tripCount = *maybeTrip;
  auto candidates = getPowerOfTwoChoices(std::min(tripCount, availBanks));

  Operation *reduceProducer = nullptr;
  if (auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    if (yield.getNumOperands() == 1)
      reduceProducer = yield.getOperand(0).getDefiningOp();
  }

  int64_t opCountPerIter = 0;
  int64_t bodyLatency = 0;
  int64_t reduceLatency = 0;
  int64_t reduceBitwidth = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isSupportedBitsSliceComputeOp(&op))
      continue;
    ++opCountPerIter;
    int64_t lat = getComputeLatency(&op);
    bodyLatency += lat;
    if (&op == reduceProducer) {
      reduceLatency = lat;
      if (op.getNumResults() > 0)
        reduceBitwidth = getSliceBitwidth(op.getResult(0));
    }
  }
  int64_t cloneLatency = kCloneCostPerBit * reduceBitwidth;

  for (int64_t regions : candidates) {
    int64_t banksPerRegion = std::max<int64_t>(1, availBanks / regions);
    int64_t iterPerRegion = std::max<int64_t>(1, tripCount / regions);
    // For reduction loops: each bank computes n iterations of pre-reduce work,
    // then folds n partial results with (n-1) intra-bank reductions before the
    // inter-bank tree reduction phase.
    bool isReduce = isSliceReductionFor(forOp) && reduceLatency > 0;
    int64_t preReduceLatency = isReduce ? (bodyLatency - reduceLatency) : bodyLatency;
    int64_t local = iterPerRegion * preReduceLatency;
    int64_t intraReduce = isReduce
        ? std::max<int64_t>(0, iterPerRegion - 1) * reduceLatency
        : 0;
    int64_t interRegionReduce = 0;
    if (isSliceReductionFor(forOp) && regions > 1)
      interRegionReduce = llvm::Log2_64(regions) * (reduceLatency + cloneLatency);
    int64_t makespan = local + intraReduce + interRegionReduce;
    if (makespan < best.estimatedMakespan) {
      best.estimatedMakespan = makespan;
      best.regionCount = regions;
      best.banksPerRegion = banksPerRegion;
      best.iterPerRegion = iterPerRegion;
      best.unrollFactor = isSliceReductionFor(forOp) ? iterPerRegion : regions;
      best.computeOpsPerRegion = iterPerRegion * opCountPerIter;
    }
  }

  for (int64_t r = 0; r < best.regionCount; ++r) {
    SmallVector<int64_t> banks;
    for (int64_t i = 0; i < best.banksPerRegion; ++i)
      banks.push_back((baseBank + r * best.banksPerRegion + i) % kNumBanks);
    best.regionBanks.push_back(std::move(banks));
  }

  return best;
}

MappingSolution modelAndSolveILP(const DAG &dag) {
  const int N = dag.nodes.size();
  int64_t totalDuration = 0;
  for (const Node &node : dag.nodes)
    totalDuration += std::max<int64_t>(1, node.duration);

  GRBEnv env = GRBEnv(true);
  env.start();
  GRBModel model = GRBModel(env);

  std::vector<std::vector<GRBVar>> x(N, std::vector<GRBVar>(kNumBanks));
  for (int i = 0; i < N; ++i) {
    for (int n = 0; n < kNumBanks; ++n) {
      x[i][n] =
          model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                       "x_" + std::to_string(i) + "_" + std::to_string(n));
    }
  }

  for (int i = 0; i < N; ++i) {
    GRBLinExpr sumBanks = 0;
    for (int n = 0; n < kNumBanks; ++n) {
      sumBanks += x[i][n];
    }
    model.addConstr(sumBanks == 1, "assign_one_bank_" + std::to_string(i));
  }

  std::vector<GRBVar> start(N), end(N);
  for (int i = 0; i < N; ++i) {
    start[i] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
                            "start_" + std::to_string(i));
    end[i] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER,
                          "end_" + std::to_string(i));
  }

  for (int i = 0; i < N; ++i) {
    model.addConstr(end[i] == start[i] + dag.nodes[i].duration,
                    "duration_" + std::to_string(i));
  }

  std::vector<std::vector<bool>> dep(N, std::vector(N, false));
  for (int i = 0; i < N; ++i) {
    for (int succ : dag.nodes[i].succs) {
      dep[i][succ] = true;
    }
  }

  for (int k = 0; k < N; ++k) {
    for (int i = 0; i < N; ++i) {
      if (dep[i][k]) {
        for (int j = 0; j < N; ++j) {
          if (dep[k][j])
            dep[i][j] = true;
        }
      }
    }
  }

  struct CloneEdge {
    int src;
    int dst;
    GRBVar active;
  };
  llvm::SmallVector<CloneEdge> cloneEdges;

  const int64_t M = std::max<int64_t>(100, totalDuration + 1024);
  for (int i = 0; i < N; ++i) {
    for (int j : dag.nodes[i].succs) {
      GRBVar cloneVar =
          model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                       "clone_" + std::to_string(i) + "_" + std::to_string(j));
      cloneEdges.push_back(CloneEdge{i, j, cloneVar});
      GRBLinExpr sumB;
      for (int k = 0; k < kNumBanks; ++k) {
        GRBVar b =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "b_" + std::to_string(i) + "_" + std::to_string(j) +
                             "_" + std::to_string(k));
        model.addConstr(b <= x[i][k]);
        model.addConstr(b <= x[j][k]);
        model.addConstr(b >= x[i][k] + x[j][k] - 1);
        sumB += b;
      }
      model.addConstr(cloneVar + sumB == 1, "clone_link_" + std::to_string(i) +
                                                "_" + std::to_string(j));
      model.addConstr(start[j] >= end[i] + kCloneCostPerBit * cloneVar,
                      "sched_dep_" + std::to_string(i) + "_" +
                          std::to_string(j));

      for (int k = 0; k < N; ++k) {
        GRBLinExpr sumBsrc;
        for (int bank = 0; bank < kNumBanks; ++bank) {
          GRBVar b = model.addVar(
              0.0, 1.0, 0.0, GRB_BINARY,
              "b_src_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                  std::to_string(k) + "_" + std::to_string(bank));
          model.addConstr(b <= x[i][bank]);
          model.addConstr(b <= x[k][bank]);
          model.addConstr(b >= x[i][bank] + x[k][bank] - 1);
          sumBsrc += b;
        }
        GRBVar sameBankSrc =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "same_src_" + std::to_string(i) + "_" +
                             std::to_string(j) + "_" + std::to_string(k));
        model.addConstr(sumBsrc == sameBankSrc,
                        "same_src_link_" + std::to_string(i) + "_" +
                            std::to_string(j) + "_" + std::to_string(k));
        GRBVar orderSrc =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "order_src_" + std::to_string(i) + "_" +
                             std::to_string(j) + "_" + std::to_string(k));
        model.addConstr(
            end[k] <= end[i] + M * (2 - cloneVar - sameBankSrc + orderSrc),
            "clone_src_before_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        model.addConstr(
            start[k] >= end[i] + kCloneCostPerBit -
                            M * (2 - cloneVar - sameBankSrc + (1 - orderSrc)),
            "clone_src_after_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));

        GRBLinExpr sumBdst;
        for (int bank = 0; bank < kNumBanks; ++bank) {
          GRBVar b = model.addVar(
              0.0, 1.0, 0.0, GRB_BINARY,
              "b_dst_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                  std::to_string(k) + "_" + std::to_string(bank));
          model.addConstr(b <= x[j][bank]);
          model.addConstr(b <= x[k][bank]);
          model.addConstr(b >= x[j][bank] + x[k][bank] - 1);
          sumBdst += b;
        }
        GRBVar sameBankDst =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "same_dst_" + std::to_string(i) + "_" +
                             std::to_string(j) + "_" + std::to_string(k));
        model.addConstr(sumBdst == sameBankDst,
                        "same_dst_link_" + std::to_string(i) + "_" +
                            std::to_string(j) + "_" + std::to_string(k));
        GRBVar orderDst =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "order_dst_" + std::to_string(i) + "_" +
                             std::to_string(j) + "_" + std::to_string(k));
        model.addConstr(
            end[k] <= end[i] + M * (2 - cloneVar - sameBankDst + orderDst),
            "clone_dst_before_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        model.addConstr(
            start[k] >= end[i] + kCloneCostPerBit -
                            M * (2 - cloneVar - sameBankDst + (1 - orderDst)),
            "clone_dst_after_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
      }
    }
  }

  for (size_t e1 = 0; e1 < cloneEdges.size(); ++e1) {
    for (size_t e2 = e1 + 1; e2 < cloneEdges.size(); ++e2) {
      const CloneEdge &c1 = cloneEdges[e1];
      const CloneEdge &c2 = cloneEdges[e2];
      auto addCloneOrderConstraints = [&](int a, int b, llvm::StringRef tag) {
        GRBLinExpr sumB;
        for (int bank = 0; bank < kNumBanks; ++bank) {
          GRBVar bVar = model.addVar(
              0.0, 1.0, 0.0, GRB_BINARY,
              "b_" + std::string(tag) + "_" + std::to_string(a) + "_" +
                  std::to_string(b) + "_" + std::to_string(bank));
          model.addConstr(bVar <= x[a][bank]);
          model.addConstr(bVar <= x[b][bank]);
          model.addConstr(bVar >= x[a][bank] + x[b][bank] - 1);
          sumB += bVar;
        }
        GRBVar sameBank =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "same_" + std::string(tag) + "_" + std::to_string(a) +
                             "_" + std::to_string(b));
        model.addConstr(sumB == sameBank, "same_link_" + std::string(tag) +
                                              "_" + std::to_string(a) + "_" +
                                              std::to_string(b));
        GRBVar orderVar =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "order_" + std::string(tag) + "_" + std::to_string(a) +
                             "_" + std::to_string(b));
        model.addConstr(end[c1.src] + kCloneCostPerBit <=
                            end[c2.src] + M * (3 - c1.active - c2.active -
                                               sameBank + orderVar),
                        "clone_serial_ab_" + std::string(tag) + "_" +
                            std::to_string(a) + "_" + std::to_string(b));
        model.addConstr(end[c2.src] + kCloneCostPerBit <=
                            end[c1.src] + M * (3 - c1.active - c2.active -
                                               sameBank + (1 - orderVar)),
                        "clone_serial_ba_" + std::string(tag) + "_" +
                            std::to_string(a) + "_" + std::to_string(b));
      };

      addCloneOrderConstraints(c1.src, c2.src, "src_src");
      addCloneOrderConstraints(c1.dst, c2.dst, "dst_dst");
      addCloneOrderConstraints(c1.src, c2.dst, "src_dst");
      addCloneOrderConstraints(c1.dst, c2.src, "dst_src");
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      if (!dep[i][j] && !dep[j][i] && !dag.nodes[i].isContainer &&
          !dag.nodes[j].isContainer) {
        GRBVar orderVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                                       "order_" + std::to_string(i) + "_" +
                                           std::to_string(j));
        GRBVar diffVar =
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                         "diff_" + std::to_string(i) + "_" + std::to_string(j));
        GRBLinExpr sumB;
        for (int k = 0; k < kNumBanks; ++k) {
          GRBVar b =
              model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                           "b_" + std::to_string(i) + "_" + std::to_string(j) +
                               "_" + std::to_string(k));
          model.addConstr(b <= x[i][k]);
          model.addConstr(b <= x[j][k]);
          model.addConstr(b >= x[i][k] + x[j][k] - 1);
          sumB += b;
        }
        model.addConstr(diffVar + sumB == 1, "diff_link_" + std::to_string(i) +
                                                 "_" + std::to_string(j));
        model.addConstr(start[j] >= end[i] - M * (diffVar + 1.0 - orderVar),
                        "serial_ij_" + std::to_string(i) + "_" +
                            std::to_string(j));
        model.addConstr(start[i] >= end[j] - M * (diffVar + orderVar),
                        "serial_ji_" + std::to_string(i) + "_" +
                            std::to_string(j));
      }
    }
  }

  GRBVar makespan =
      model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, "makespan");
  for (int i = 0; i < N; ++i) {
    if (dag.nodes[i].isContainer)
      continue;
    model.addConstr(makespan >= end[i], "mkspan_ge_end_" + std::to_string(i));
  }

  model.setObjective(makespan + 0, GRB_MINIMIZE);
  model.optimize();

  int status = model.get(GRB_IntAttr_Status);
  if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL) {
    std::cerr << "Solving ended with status: " << status << "\n";
    std::vector<int> fallbackChosen(N, -1);
    std::vector<int64_t> fallbackStart(N, 0);
    std::vector<int64_t> fallbackEnd(N, 0);
    std::vector<int64_t> bankCursor(kNumBanks, 0);
    for (int i = 0; i < N; ++i) {
      int bank = i % kNumBanks;
      int64_t start = bankCursor[bank];
      int64_t end = start + dag.nodes[i].duration;
      fallbackChosen[i] = bank;
      fallbackStart[i] = start;
      fallbackEnd[i] = end;
      bankCursor[bank] = end;
    }
    return MappingSolution{fallbackChosen, fallbackStart, fallbackEnd};
  }

  // std::vector<std::pair<int, int>> chosen;
  // chosen.reserve(N);
  std::vector<int> chosen(N, -1);
  std::vector<int64_t> startChosen(N, 0);
  std::vector<int64_t> endChosen(N, 0);

  for (int i = 0; i < N; ++i) {
    for (int n = 0; n < kNumBanks; ++n) {
      double val = x[i][n].get(GRB_DoubleAttr_X);
      if (val > 0.5) {
        // chosen.emplace_back(i, n);
        chosen[i] = n;
        break;
      }
    }
    startChosen[i] = static_cast<int64_t>(start[i].get(GRB_DoubleAttr_X));
    endChosen[i] = static_cast<int64_t>(end[i].get(GRB_DoubleAttr_X));
  }

  return MappingSolution{chosen, startChosen, endChosen};
}

struct BitsOptimiseMappingPass
    : public impl::BitsOptmiseMappingPassBase<BitsOptimiseMappingPass> {
  using Base::Base;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    DAG dag;
    DenseMap<Operation *, int64_t> loopTripCount;
    DenseMap<Operation *, int64_t> loopBestBanks;

    auto collectNodes = [&](auto &&self, Region &region, int parentForNode,
                            int64_t loopDepth) -> LogicalResult {
      for (Operation &op : region.getOps()) {
        if (isSupportedBitsSliceComputeOp(&op)) {
          int id = dag.nodes.size();
          dag.nodes.push_back(
              Node{&op, getComputeLatency(&op), false, parentForNode, loopDepth,
                   {}, {}});
          dag.opToNodeID[&op] = id;
          continue;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          FailureOr<int64_t> tripCount = getConstTripCount(forOp);
          if (failed(tripCount) || !llvm::isPowerOf2_64(*tripCount)) {
            forOp.emitOpError("requires constant power-of-two trip count");
            return failure();
          }
          bool hasSliceReduction = isSliceReductionFor(forOp);
          bool nestedFor = hasNestedForOp(forOp);
          bool spansMultipleBankWaves = *tripCount > kNumBanks;
          bool aggregateLoopWork =
              hasSliceReduction || nestedFor || spansMultipleBankWaves;
          int64_t estLatency = aggregateLoopWork ? estimateForLatency(forOp) : 0;
          int id = dag.nodes.size();
          dag.nodes.push_back(Node{forOp.getOperation(), estLatency,
                                   !aggregateLoopWork, parentForNode, loopDepth,
                                   {}, {}});
          dag.opToNodeID[&op] = id;
          loopTripCount[&op] = *tripCount;
          loopBestBanks[&op] = estimateBestParallelBanksForLoop(forOp);
          // Only recurse into the loop body when the loop is a transparent
          // container (isContainer = true, i.e. !aggregateLoopWork). For
          // aggregate loops the estimated duration already subsumes the body;
          // adding body nodes as independent ILP tasks would allow the solver
          // to schedule them in parallel with their own enclosing loop, which
          // is physically incorrect. Bank assignment for body ops inside
          // aggregate loops is handled by the for-loop planning phase (Phase 5).
          if (!aggregateLoopWork) {
            if (failed(self(self, forOp.getRegion(), id, loopDepth + 1)))
              return failure();
          }
        }
      }
      return success();
    };

    if (failed(collectNodes(collectNodes, func.getBody(), -1, 0))) {
      signalPassFailure();
      return;
    }

    if (dag.nodes.empty()) {
      return;
    }

    for (int id = 0; id < (int)dag.nodes.size(); ++id) {
      Operation *curOp = dag.nodes[id].op;
      for (Value operand : curOp->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          auto it = dag.opToNodeID.find(defOp);
          if (it == dag.opToNodeID.end()) {
            continue;
          }

          int predID = it->second;
          if (kEnableVerboseMappingLog)
            std::cout << predID << " -> " << id << "\n";
          dag.nodes[predID].succs.push_back(id);
          dag.nodes[id].preds.push_back(predID);
        }
      }
    }

    try {
      MappingSolution solution = modelAndSolveILP(dag);
      const std::vector<int> &chosen = solution.chosenBank;
      MLIRContext *ctx = func.getContext();
      Builder attrBuilder(ctx);
 
      // Phase 4: annotate ILP nodes.
      // Attributes set on each compute op:
      //   bits.bank_id    i64  – ILP-assigned bank (also set on scf.for nodes)
      //   bits.start_time i64  – cycle at which execution begins
      //   bits.end_time   i64  – cycle at which the result is ready
      //   bits.out_xfer   array – one entry per cross-bank consumer:
      //                     { result_idx i64, dst_bank i64,
      //                       send_time i64, recv_time i64 }
      //                     send_time = end_time of this op
      //                     recv_time = start_time of the consumer op
      //   bits.in_xfer    array – one entry per cross-bank operand:
      //                     { src_bank i64, operand_idx i64, ready_time i64 }
      //                     ready_time = start_time of this op
      DenseMap<Operation *, SmallVector<Attribute>> inXferMap;
      for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
        if (chosen[i] < 0) {
          continue;
        }
        Operation *op = dag.nodes[i].op;
        op->setAttr("bits.bank_id",
                    IntegerAttr::get(IntegerType::get(ctx, 64), chosen[i]));
 
        if (isa<scf::ForOp>(op))
          continue;
 
        op->setAttr("bits.start_time",
                    IntegerAttr::get(IntegerType::get(ctx, 64),
                                     solution.startTime[i]));
        op->setAttr("bits.end_time",
                    IntegerAttr::get(IntegerType::get(ctx, 64),
                                     solution.endTime[i]));

        SmallVector<Attribute> outXfers;
        for (auto [resultIdx, result] : llvm::enumerate(op->getResults())) {
          for (Operation *user : result.getUsers()) {
            auto it = dag.opToNodeID.find(user);
            if (it == dag.opToNodeID.end())
              continue;
            int j = it->second;
            if (chosen[j] < 0 || chosen[j] == chosen[i])
              continue;
            // Cross-bank transfer: build out_xfer entry on the source op.
            NamedAttrList xfer;
            xfer.set("result_idx",
                     attrBuilder.getI64IntegerAttr((int64_t)resultIdx));
            xfer.set("dst_bank",
                     IntegerAttr::get(IntegerType::get(ctx, 64), chosen[j]));
            xfer.set("send_time",
                     IntegerAttr::get(IntegerType::get(ctx, 64),
                                      solution.endTime[i]));
            xfer.set("recv_time",
                     IntegerAttr::get(IntegerType::get(ctx, 64),
                                      solution.startTime[j]));
            outXfers.push_back(DictionaryAttr::get(ctx, xfer.getAttrs()));
            // Build in_xfer entry on each matching operand slot of the consumer.
            for (auto [opIdx, operand] : llvm::enumerate(user->getOperands())) {
              if (operand.getDefiningOp() != op)
                continue;
              NamedAttrList inXfer;
              inXfer.set("src_bank",
                         IntegerAttr::get(IntegerType::get(ctx, 64), chosen[i]));
              inXfer.set("operand_idx",
                         attrBuilder.getI64IntegerAttr((int64_t)opIdx));
              inXfer.set("ready_time",
                         IntegerAttr::get(IntegerType::get(ctx, 64),
                                          solution.startTime[j]));
              inXferMap[user].push_back(DictionaryAttr::get(ctx, inXfer.getAttrs()));
            }
          }
        }
        if (!outXfers.empty())
          op->setAttr("bits.out_xfer", ArrayAttr::get(ctx, outXfers));
      }
      // Apply accumulated in_xfer annotations.
      for (auto &[destOp, inXfers] : inXferMap)
        destOp->setAttr("bits.in_xfer", ArrayAttr::get(ctx, inXfers));
 
      // Phase 5: for-loop parallelism planning.
      // Attributes set on each scf.for:
      //   bits.parallel_factor  i64 – number of banks sharing the loop
      //   bits.loop_kind        str – "reduction" | "parallel"
      //   bits.iters_per_bank   i64 – tripCount / parallel_factor
      //
      // Additional attributes for reduction loops only:
      //   bits.intra_reduce_count i64   – (iters_per_bank - 1) per-bank reductions
      //   bits.reduce_levels      i64   – log2(parallel_factor) inter-bank levels
      //   bits.reduce_tree  array<array<{src_bank i64, dst_bank i64}>>
      //                           one inner array per tree level (outermost first);
      //                           src sends to dst at each step
      //
      // The reduce op inside each reduction loop body also receives:
      //   bits.is_reduce_op (unit) – distinguishes compute from reduction steps
      // Helper: apply all loop scheduling attributes to a for op.
      auto annotateForLoop = [&](scf::ForOp lp, int64_t parallelFactor,
                                 int64_t tc, bool isReduction) {
        lp->setAttr("bits.parallel_factor",
                    attrBuilder.getI64IntegerAttr(parallelFactor));
        lp->setAttr("bits.loop_kind",
                    StringAttr::get(ctx, isReduction ? "reduction" : "parallel"));
        int64_t itersPerBank =
            parallelFactor > 0 ? tc / parallelFactor : tc;
        lp->setAttr("bits.iters_per_bank",
                    attrBuilder.getI64IntegerAttr(itersPerBank));
        if (!isReduction)
          return;
        lp->setAttr("bits.intra_reduce_count",
                    attrBuilder.getI64IntegerAttr(
                        std::max<int64_t>(0, itersPerBank - 1)));
        int64_t reduceLevels =
            parallelFactor > 1 ? (int64_t)llvm::Log2_64(parallelFactor) : 0;
        lp->setAttr("bits.reduce_levels",
                    attrBuilder.getI64IntegerAttr(reduceLevels));
        // Build inter-bank tree descriptor using the ILP-assigned base bank.
        // At each level the "right half" of active banks sends to the "left
        // half"; only the left banks survive to the next level.
        int64_t baseBank = 0;
        if (auto bankAttr = lp->getAttrOfType<IntegerAttr>("bits.bank_id"))
          baseBank = bankAttr.getInt();
        SmallVector<Attribute> treeLevels;
        for (int64_t step = parallelFactor / 2; step >= 1; step /= 2) {
          SmallVector<Attribute> levelPairs;
          for (int64_t idx = 0; idx < step; ++idx) {
            int64_t dstBank = (baseBank + idx) % kNumBanks;
            int64_t srcBank = (baseBank + idx + step) % kNumBanks;
            NamedAttrList pair;
            pair.set("src_bank",
                     IntegerAttr::get(IntegerType::get(ctx, 64), srcBank));
            pair.set("dst_bank",
                     IntegerAttr::get(IntegerType::get(ctx, 64), dstBank));
            levelPairs.push_back(DictionaryAttr::get(ctx, pair.getAttrs()));
          }
          treeLevels.push_back(ArrayAttr::get(ctx, levelPairs));
        }
        if (!treeLevels.empty())
          lp->setAttr("bits.reduce_tree", ArrayAttr::get(ctx, treeLevels));
      };
      SmallVector<scf::ForOp> allForOps;
      func.walk([&](scf::ForOp forOp) { allForOps.push_back(forOp); });

      for (scf::ForOp forOp : allForOps) {
        if (forOp->hasAttr("bits.parallel_factor"))
          continue;

        auto tripMaybe = getConstTripCount(forOp);
        if (failed(tripMaybe))
          continue;
        int64_t tripCount = *tripMaybe;

        scf::ForOp childFor;
        int childForCount = 0;
        for (Operation &op : forOp.getBody()->without_terminator()) {
          if (auto nested = dyn_cast<scf::ForOp>(&op)) {
            childFor = nested;
            ++childForCount;
          }
        }

        if (childForCount == 1) {
          int64_t bestCost = std::numeric_limits<int64_t>::max();
          int64_t bestOuterRegions = 1;
          int64_t bestInnerParallelFactor = 1;
 
          for (int64_t regions :
               getPowerOfTwoChoices(std::min<int64_t>(tripCount, kNumBanks))) {
            int64_t banksPerRegion = std::max<int64_t>(1, kNumBanks / regions);
            RegionPlan innerPlan =
                planLeafForRegions(childFor, 0, banksPerRegion);
            int64_t iterPerRegion = std::max<int64_t>(1, tripCount / regions);
            int64_t cost = iterPerRegion * innerPlan.estimatedMakespan;
            if (cost < bestCost) {
              bestCost = cost;
              bestOuterRegions = regions;
              bestInnerParallelFactor = innerPlan.regionCount;
            }
          }
          // Outer loop is always parallel (instances are independent).
          annotateForLoop(forOp, bestOuterRegions, tripCount,
                          /*isReduction=*/false);
          auto childTripMaybe = getConstTripCount(childFor);
          int64_t childTripCount =
              succeeded(childTripMaybe) ? *childTripMaybe : 1;
          annotateForLoop(childFor, bestInnerParallelFactor, childTripCount,
                          isSliceReductionFor(childFor));

        } else {
          RegionPlan plan = planLeafForRegions(forOp, 0, kNumBanks);
          annotateForLoop(forOp, plan.regionCount, tripCount,
                          isSliceReductionFor(forOp));
        }
      }
      // Mark the accumulator op inside each slice-reduction loop body so the
      // dispatcher can distinguish regular compute work from reduction steps.
      func.walk([&](scf::ForOp forOp) {
        if (!isSliceReductionFor(forOp))
          return;
        auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
          return;
        Operation *reduceProducer = yield.getOperand(0).getDefiningOp();
        if (reduceProducer && isSupportedBitsSliceComputeOp(reduceProducer))
          reduceProducer->setAttr("bits.is_reduce_op", UnitAttr::get(ctx));
      });

      dumpFlattenedEventSequence(func);
    } catch (GRBException &e) {
      std::cerr << "Gurobi error: " << e.getMessage() << "\n";
    } catch (std::exception &ex) {
      std::cerr << "Error: " << ex.what() << "\n";
    }
  }
};

} // namespace mlir::bits
