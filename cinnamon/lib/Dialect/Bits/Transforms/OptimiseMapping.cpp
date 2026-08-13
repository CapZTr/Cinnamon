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

// Configured from the pass options (num-banks / mem-type) at runOnOperation
// entry; the pass runs on a single func so plain globals are safe here.
// Time unit throughout: one AAP.
// Largest loop body (in mapper-visible compute ops) that may be lifted into
// the ILP as individual nodes; see the use site in collectNodes.
static constexpr int64_t kMaxIlpNodesPerLoopBody = 16;

static int64_t kNumBanks = 32;
static int64_t kBanksPerChannel = 32;        // ddr: 32, hbm: 16
static int64_t kCopyCostIntraChPerBit = 19;  // same-channel inter-bank; hbm: 6
static int64_t kCopyCostCrossChPerBit = 10;  // cross-channel;           hbm: 3

// Which cost model the mapper optimises against (pass option cost-model).
//
// Exact: every op is priced at the row-op count its gem5 expansion actually
// emits, and a row copy at a cost that depends on where its two banks sit.
//
// Proxy: one time unit per bit-plane row, for every op and every copy alike.
// This is the model one writes down before measuring anything -- "a bit-serial
// op touches one row per bit, so it costs its bitwidth" -- and it is wrong in
// two independent ways that this pass exists to exploit: real per-bit costs are
// neither 1 nor equal across ops (a bit-serial multiply is quadratic in the
// operand widths, an add linear), and a row copy's cost depends on whether it
// crosses a channel. Selecting Proxy changes only the mapper's DECISIONS
// (parallel bank counts, reduce-tree shapes, ILP placement); the row-op
// sequence emitted for a given decision is unchanged, so a proxy-mapped trace
// asks the simulator to do exactly the same work in a differently chosen
// schedule. That is what makes the two comparable.
enum class CostModel { Exact, Proxy };
static CostModel kCostModel = CostModel::Exact;

// The four things this cost model knows that a naive one does not. Each can be
// switched off independently (pass option proxy-blind) so the ablation can
// attribute a slowdown to a named modelling feature instead of to an opaque
// "bad model", and so the strawman is a documented position rather than a
// hand-tuned one.
//
//   oplevel   OFF: every op costs one unit per bit-plane row it touches.
//             ON : the exact row-op count of that op's expansion on that
//                  substrate (a bit-serial multiply is quadratic in the operand
//                  widths, an add linear, a range scan linear with a different
//                  slope -- none of them one-per-row).
//   width     OFF: an op is one unit and a row copy is one unit, full stop --
//                  the bitwidth does not enter at all.  Implies oplevel off.
//   location  OFF: every inter-bank copy costs the same wherever its two banks
//                  sit, so no pairing is preferable and the reduce tree is
//                  plain stride halving over the flat bank list.
//             ON : intra- vs cross-channel pricing + the channel-aware tree.
//   pack      OFF: each loop iteration is priced as its own full-row wave.
//             ON : kBankRowWidth lanes of a leaf reduction share one row, so a
//                  bank runs ceil(iters/P) waves and then folds P lanes.
//
// pack is the one with history: the estimator was pack-blind while the emitter
// was pack-aware, and conv1x1's predicted makespan came out independent of the
// bank count (16x off). So it is a real error mode, not an invented one.
static bool kBlindOpLevel = false;   // true = op-level costs are NOT modelled
static bool kBlindWidth = false;
static bool kBlindLocation = false;
static bool kBlindPack = false;

// Bitwidth as the COST MODEL sees it. Only ever used to price things -- the
// bitwidth written into a trace record is always the true one, since that is
// the work the simulator must perform.
static int64_t costWidth(int64_t bw) { return kBlindWidth ? 1 : bw; }

// Which PuD substrate the exact costs are calibrated against (pass option
// backend). Every formula below is the row-op count that substrate's gem5
// expansion actually emits, verified packet-for-packet -- the two differ per
// op and not by a constant factor (PRADA's sequential row activation gives it a
// cheaper full adder and multiplier, while the absence of dual-contact cells
// makes its XNOR and range scan dearer), so a schedule optimised for one is
// measurably not optimal for the other. Mapping for PRADA with SIMDRAM's
// numbers is exactly the mis-costing this pass's cost-model ablation is about.
enum class Backend { Simdram, Prada };
static Backend kBackend = Backend::Simdram;

// Emit plain bits.add_i as a WIDENING add (pass option wide-add): same
// operands, one extra row-op to store the carry-out, and a result one bit wider
// -- so the sum cannot silently wrap. Both the price and the emitted record
// kind move together; see makeAddiEvent.
static bool kWideAdd = false;
// Intra-bank lane-fold copy: one AAP per bit-plane row, matching how the
// gem5 RowOpTracePlayer replays src==dst ROW_COPY records (same-subarray
// RowClone FPM path, tRAS+tWLOV+tRP per row). NOTE this is a deliberate
// idealisation kept consistent across scheduler and simulator: physically,
// folding P packed SIMD lanes needs a COLUMN-shifted copy (the surviving
// lanes must move sideways to align with their partners), which neither a
// plain AAP nor any modelled gem5 primitive performs.
constexpr int64_t kLaneFoldCostPerBit = 1;

static int64_t channelOfBank(int64_t bank) { return bank / kBanksPerChannel; }

// Per-bit cost of one reduce-tree row copy, by physical location. Calibrated
// against gem5 (rowOpMakespan steady state, in units of one AAP): the
// intra-channel copy streams the row twice over one channel data bus, the
// cross-channel copy streams it once per channel over two buses that
// pipeline. src == dst keeps the lane-fold proxy.
static int64_t copyCostPerBit(int64_t srcBank, int64_t dstBank) {
  if (srcBank == dstBank)
    return kLaneFoldCostPerBit;
  return channelOfBank(srcBank) == channelOfBank(dstBank)
             ? kCopyCostIntraChPerBit
             : kCopyCostCrossChPerBit;
}
// Width (in vector lanes / columns) of one bank row. Independent same-kind,
// same-bitwidth ops whose vector lengths sum to <= this value can be packed
// into a single bank row and executed as one concurrent SIMD wave.
constexpr int64_t kBankRowWidth = 8192;
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
  enum class Kind { Muli, Addi, RowCopy, Relu, AddiWide, Xnor, RangeScan,
                    Min, Max, Subi };
  Kind kind = Kind::Muli;
  int64_t start = 0;
  int64_t end = 0;
  MuliEvent muli;
  AddiEvent addi;
  AddiEvent subi;   // same payload shape: two equal-width operands
  RowCopyEvent rowCopy;
  // A ReLU record carries the same fields as an ADDI one (bitwidth + bank
  // mask), so it reuses AddiEvent rather than duplicating the struct.
  AddiEvent relu;
  // Widening add: lhsBitwidth is the OPERAND width; the result is one row
  // wider. Same field shape as an ADDI record, so it reuses AddiEvent.
  AddiEvent addiWide;
  // Bitwise XNOR (the BNN multiply): elementwise, N x N -> N. Same field
  // shape as an ADDI record, so it reuses AddiEvent.
  AddiEvent xnorOp;
  // BitWeaving range scan: lhsBitwidth is the SCANNED column width; the
  // result is always a one-bit mask. Same field shape as an ADDI record.
  AddiEvent rangeScan;
  // Bit-serial min/max: elementwise, N x N -> N. Same field shape as an ADDI
  // record; the Kind distinguishes which of the two it is.
  AddiEvent minMax;
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

static int64_t getSliceVectorLength(Value value) {
  auto sliceTy = dyn_cast<SliceType>(value.getType());
  return sliceTy ? sliceTy.getVectorLength() : 0;
}

// Identifies whether two ops may share a bank row and run concurrently. Only
// add/mul are packable, and two ops pack only when their signatures match
// exactly (same kind and same operand bitwidths) so that they have identical
// compute latency and execute in lockstep. `kind` < 0 means "not packable".
struct PackSignature {
  int kind = -1; // 0 = bits.addi, 1 = bits.muli
  int64_t lhsBitwidth = 0;
  int64_t rhsBitwidth = 0;
  bool packable() const { return kind >= 0; }
  bool operator==(const PackSignature &o) const {
    return kind == o.kind && lhsBitwidth == o.lhsBitwidth &&
           rhsBitwidth == o.rhsBitwidth;
  }
};

static PackSignature getPackSignature(Operation *op) {
  if (auto add = dyn_cast<AddIOp>(op))
    return {0, getSliceBitwidth(add.getLhs()), getSliceBitwidth(add.getRhs())};
  if (auto mul = dyn_cast<MulIOp>(op))
    return {1, getSliceBitwidth(mul.getLhs()), getSliceBitwidth(mul.getRhs())};
  return {};
}

static int64_t getComputeLatency(Operation *op) {
  // Width-blind: one unit per op, the bitwidth does not enter at all.
  if (kBlindWidth)
    return op->getNumResults() == 0 ? 0 : 1;
  // Op-level-blind: one time unit per bit-plane row the op processes, for every
  // op alike. "Rows processed" is the widest slice the op touches, which is the
  // result for an add/multiply/min/max/xnor and the INPUT for a range scan
  // (whose result is a single mask row and says nothing about the work). Taking
  // the max over operands and results expresses that uniformly, with no per-op
  // case analysis -- having none is the point of this model.
  if (kBlindOpLevel) {
    int64_t rows = 0;
    for (Value v : op->getOperands())
      rows = std::max(rows, getSliceBitwidth(v));
    for (Value r : op->getResults())
      rows = std::max(rows, getSliceBitwidth(r));
    return rows;
  }
  // Exact row-op counts of the gem5 expansion for the selected backend (AAP
  // units). Every entry below was checked against a measured packet count, not
  // read off a paper:
  //
  //                        SIMDRAM (Ambit AAP/AP)     PRADA (TRA / N / 5RA)
  //   ADDI(n)              8n + 1                     6n + 1
  //   ADDI_WIDE(n)         8n + 2                     6n + 2
  //   MULI(m,n)            13mn + 4m - 8              10mn + 4m - 5
  //   AND(n) / OR(n)       4n                         4n
  //   ReLU-masked AND(n)   4n + 2                     4n + 1
  //   XNOR(n)              7n                         10n
  //   RANGE_SCAN(L)        4L + 5                     6L + 6
  //   MIN/MAX(n)           13n + 2                    13n + 1
  //
  // Where they differ, it is a property of the substrate and not a fudge
  // factor: PRADA gets a 3-row majority in ONE command (so its full adder is
  // 6 row-ops against Ambit's 9, and its multiplier follows), but it has no
  // dual-contact cell, so every complement costs an explicit NOT -- which is
  // why its XNOR (two complements per bit) and its range scan (which rebuilds
  // ~v every bit) are dearer. Note the two are not related by any constant
  // factor, which is exactly why a mapping decided under one substrate's costs
  // is not the mapping the other one wants.
  const bool prada = kBackend == Backend::Prada;
  if (auto add = dyn_cast<AddIOp>(op)) {
    // With wide-add on, this op is emitted as the widening add and must be
    // priced as one: the carry-out store is one extra row-op.
    int64_t perBit = prada ? 6 : 8;
    return perBit * getSliceBitwidth(add.getResult()) + (kWideAdd ? 2 : 1);
  }
  // Subtraction reuses the full-adder chain with the carry initialised to one
  // and the subtrahend complemented. The adder reads the subtrahend row twice
  // per bit, so the complement must be materialised rather than read on the fly:
  // one extra AAP per bit through a dual-contact cell on SIMDRAM (9n + 1), one
  // extra single-command NOT per bit on PRADA (7n + 1).
  if (auto sub = dyn_cast<SubIOp>(op))
    return (prada ? 7 : 9) * getSliceBitwidth(sub.getResult()) + 1;
  if (auto mul = dyn_cast<MulIOp>(op)) {
    int64_t m = getSliceBitwidth(mul.getLhs());
    int64_t n = getSliceBitwidth(mul.getRhs());
    // shift-add over the same schedule on both backends; only the per-bit AND
    // and full-adder underneath it change.
    return prada ? 10 * m * n + 4 * m - 5 : 13 * m * n + 4 * m - 8;
  }
  // Widening add: 1 carry init + one full adder per bit + 1 carry-out store.
  // Charging the operand width (not the result width) is the whole point --
  // it is what makes this cheaper than pre-extending both operands and using
  // a truncating bits.add_i, which would cost 8*(n+1) + 1 (SIMDRAM).
  if (auto addFull = dyn_cast<AddIFullOp>(op))
    return (prada ? 6 : 8) * getSliceBitwidth(addFull.getLhs()) + 2;
  if (auto andOp = dyn_cast<AndOp>(op)) {
    // A ReLU-style masked AND additionally inverts the sign row once: two AAP
    // through the dual-contact cell on SIMDRAM (write DCC1, read DCC1N), a
    // single-command NOT on PRADA. gem5's expandRelu emits exactly that.
    int64_t base = 4 * getSliceBitwidth(andOp.getResult());
    if (!op->hasAttr("bits.masked_zero"))
      return base;
    return base + (prada ? 1 : 2);
  }
  if (auto orOp = dyn_cast<OrOp>(op))
    return 4 * getSliceBitwidth(orOp.getResult());
  // XNOR (the BNN multiply) = (a|~b) & (~a|b): three majority gates either way.
  // On SIMDRAM the dual-contact cells hand back both complements for free, so
  // expandXnor needs only 7 commands per bit (5 AAP + 2 AP); PRADA spends two
  // NOTs and has to re-stage operands between majorities, costing 10. XOR is
  // the same sequence with the two constant control rows swapped, so it would
  // cost the same -- NOT the 3n an earlier draft of this comment guessed.
  // bits.xor has no record kind today, so it is deliberately left on the
  // fallback below rather than given a cost it cannot honour.
  if (auto xnorOp = dyn_cast<XNOrOp>(op))
    return (prada ? 10 : 7) * getSliceBitwidth(xnorOp.getResult());
  // BitWeaving range scan: per scanned bit plane, load the plane, write one
  // branch-dependent selector row and advance each of the two comparison
  // chains by one majority -- 4 commands on SIMDRAM, 6 on PRADA (which needs
  // ~v rebuilt and both operands staged per bit), plus chain seeding and mask
  // finalisation. Priced off the INPUT width -- the result is a one-bit mask,
  // so the result width says nothing about the work. The scanned constants do
  // not appear: every branch of the per-bit case analysis emits the same
  // command count.
  if (auto scan = dyn_cast<RangeScanOp>(op)) {
    int64_t L = getSliceBitwidth(scan.getInput());
    return prada ? 6 * L + 6 : 4 * L + 5;
  }
  // Bit-serial min/max = compare then select. The comparator is one majority
  // gate per bit (3 commands on either substrate) and a 3-majority mux selects
  // per bit (10), so both land on 13 per bit; they differ only in seeding the
  // verdict row. Both ops cost the same -- they differ only in which operand
  // each mux branch takes.
  if (isa<MaxOp>(op) || isa<MinOp>(op))
    return 13 * getSliceBitwidth(op->getResult(0)) + (prada ? 1 : 2);

  if (op->getNumResults() == 0)
    return 0;
  return 6 * getSliceBitwidth(op->getResult(0));
}

// Build the flattened event for a plain bits.add_i. With wide-add on the op is
// emitted as the widening add instead -- kind ADDI_WIDE, which the simulator
// expands to the same full adders plus one row-op storing the carry-out, so the
// result is one bit wider and the sum cannot wrap.
//
// This is a helper and not four copies of the same conditional on purpose: a
// reduction's combining add is emitted at THREE separate sites (per-wave
// accumulate, intra-bank fold, cross-bank tree) besides the linear-op path, and
// an arm missing at any one of them changes the emitted trace silently.
static FlattenedEvent makeAddiEvent(AddIOp add, int64_t start, int64_t end,
                                    int64_t bank) {
  FlattenedEvent evt;
  evt.kind = kWideAdd ? FlattenedEvent::Kind::AddiWide
                      : FlattenedEvent::Kind::Addi;
  evt.start = start;
  evt.end = end;
  AddiEvent &payload = kWideAdd ? evt.addiWide : evt.addi;
  payload.lhsBitwidth = getSliceBitwidth(add.getLhs());
  payload.rhsBitwidth = getSliceBitwidth(add.getRhs());
  payload.start = start;
  payload.end = end;
  payload.banks.push_back(bank);
  return evt;
}

// ---------------------------------------------------------------------------
// Lossless reduction widths.
//
// A reduction of n values of `base` bits does not fit in `base` bits: it needs
// base + ceil(log2 n). The IR does not say so -- `linalg.matvec` accumulates
// into the output element type, so the emitted `bits.add_i` chain is a WRAPPING
// accumulation at a fixed width, and pricing every combine at that one width is
// correct for what the IR asks for.
//
// `wide-add` is the ablation arm that asks what it costs NOT to wrap, and for
// that the width has to travel along the reduction: the k-th combine operates on
// however many bits the partial sum has grown to by then, and so does every row
// copy that moves a partial sum between banks. Charging a flat +1 AAP per
// combine at the original width -- which is what this used to do -- understates
// that by more than an order of magnitude (GEMV: +0.14% against +2.5%), because
// it keeps a carry-out that nothing downstream ever consumes.
//
// The growth is logarithmic, not one bit per combine. A bit is only added when
// the running item count crosses a power of two, so a long accumulation chain
// widens by log2(n) in total, not by n-1. Combines that do not cross a boundary
// stay plain `bits.add_i` at the current width (8w + 1); the ones that do become
// `bits.add_i_full` (8w + 2), matching gem5's OP_ADDI_WIDE, which is equal-width
// k + k -> k+1 and takes its width from lhs_bw. Narrower addends are widened
// with the reserved constant-zero row, which is free.
// Width the addends of a reduction naturally occupy, which is where a lossless
// widening has to start counting from.
//
// A frontend may materialise a narrow partial result at the accumulator's
// declared width before feeding it in: BNN's blocks produce 6-bit counts that
// are carried in a 13-bit chain. Taking the declared 13 bits as the base would
// widen from there and charge for bits the sum never needs -- 128 six-bit counts
// still fit in 13, not 20. Peeling the extension recovers the real base.
// Only consulted under wide-add, so the exact path is bit-for-bit unaffected.
static int64_t reduceAddendWidth(Operation *reduceProducer, int64_t declared) {
  if (!reduceProducer)
    return declared;
  for (Value v : reduceProducer->getOperands())
    if (auto *def = v.getDefiningOp())
      if (isa<ExtensionIOp>(def) && def->getNumOperands() >= 1)
        return std::min(declared, getSliceBitwidth(def->getOperand(0)));
  return declared;
}

static int64_t losslessReduceWidth(int64_t baseWidth, int64_t items) {
  if (!kWideAdd || items <= 1)
    return baseWidth;
  return baseWidth + (int64_t)llvm::Log2_64_Ceil(items);
}

// Latency of one reduction combine on `width`-bit partial sums. `grows` says
// whether this combine is the one that pushes the sum into an extra bit, i.e.
// whether it must keep its carry-out.
static int64_t reduceAddLatencyForWidth(int64_t width, bool grows) {
  int64_t perBit = (kBackend == Backend::Prada) ? 6 : 8;
  return perBit * std::max<int64_t>(1, width) + (grows ? 2 : 1);
}

// Latency of ONE lossless reduction combine, the one that takes the running
// partial sum from `itemsBefore` combined values to `itemsAfter`. `flat` is the
// width-independent latency to fall back on: when wide-add is off (the IR's own
// wrapping accumulation is what gets emitted) or when the combiner is not an add
// -- max/min/mul reductions never widen, a max of n values is as wide as its
// inputs.
//
// This is the single place the cost model and the event emitter agree on how a
// lossless reduction widens; both call it, which is what keeps the plan the
// mapper prices and the trace it emits describing the same computation.
static int64_t reduceCombineLatency(int64_t baseWidth, int64_t itemsBefore,
                                    int64_t itemsAfter, bool isAdd,
                                    int64_t flat) {
  if (!kWideAdd || !isAdd)
    return flat;
  int64_t wIn = losslessReduceWidth(baseWidth, itemsBefore);
  int64_t wOut = losslessReduceWidth(baseWidth, itemsAfter);
  return reduceAddLatencyForWidth(wIn, wOut > wIn);
}

// Cost of the (waves - 1) wave-to-wave combines inside one bank: a linear
// accumulation, so the item count grows by one per step and the width by
// ceil(log2) of it.
static int64_t losslessChainCost(int64_t baseWidth, int64_t waves, bool isAdd,
                                 int64_t flat) {
  if (waves <= 1)
    return 0;
  if (!kWideAdd || !isAdd)
    return (waves - 1) * flat;
  int64_t total = 0;
  for (int64_t j = 1; j < waves; ++j)
    total += reduceCombineLatency(baseWidth, j, j + 1, isAdd, flat);
  return total;
}

// Cost of the intra-bank fold of the packed lanes: `levels` doublings starting
// from `waves` items, each a row copy of the partial sum plus one combine. The
// copy widens with the sum -- moving a 27-bit partial sum costs 27 rows, not the
// 16 the IR type would suggest.
static int64_t losslessFoldCost(int64_t baseWidth, int64_t waves,
                                int64_t levels, bool isAdd, int64_t flat) {
  int64_t total = 0;
  int64_t items = std::max<int64_t>(1, waves);
  for (int64_t l = 0; l < levels; ++l) {
    int64_t w = losslessReduceWidth(baseWidth, items);
    total += kLaneFoldCostPerBit * costWidth(std::max<int64_t>(1, w));
    total += reduceCombineLatency(baseWidth, items, items * 2, isAdd, flat);
    items *= 2;
  }
  return total;
}

// One reduction combine, priced and tagged at the running partial-sum width.
// With wide-add off this degenerates to exactly what makeAddiEvent produced
// before -- same kind, same widths -- so no golden trace moves.
static FlattenedEvent makeReduceAddEvent(AddIOp add, int64_t start, int64_t end,
                                         int64_t bank, int64_t width,
                                         bool grows) {
  if (!kWideAdd)
    return makeAddiEvent(add, start, end, bank);
  FlattenedEvent evt;
  evt.kind = grows ? FlattenedEvent::Kind::AddiWide : FlattenedEvent::Kind::Addi;
  evt.start = start;
  evt.end = end;
  AddiEvent &payload = grows ? evt.addiWide : evt.addi;
  payload.lhsBitwidth = width;
  payload.rhsBitwidth = width;
  payload.start = start;
  payload.end = end;
  payload.banks.push_back(bank);
  return evt;
}

static FailureOr<int64_t> getConstTripCount(scf::ForOp forOp);
static SmallVector<int64_t> buildBankGroup(int64_t baseBank, int64_t numBanks);
static SmallVector<SmallVector<std::pair<int64_t, int64_t>>>
buildChannelAwareReduceTree(ArrayRef<int64_t> activeBanks);
static int64_t estimateReduceTreeCost(int64_t banks, int64_t bitwidth,
                                      int64_t reduceLatency,
                                      int64_t itemsAtLeaf = 1,
                                      bool isAddReduce = false);
static bool isSliceReductionFor(scf::ForOp forOp);
static int64_t estimatePeakBanksPerIteration(scf::ForOp forOp);
static int64_t estimateBestParallelBanksForLoop(scf::ForOp forOp);
static int64_t getLoopPackFactor(scf::ForOp forOp);
// Pack factor as the COST MODEL sees it (see kBlindPack). Purely predictive:
// the flattener and the loop annotation must use the TRUE factor, or the trace
// would stop describing the same computation.
static int64_t getCostPackFactor(scf::ForOp forOp) {
  return kBlindPack ? 1 : getLoopPackFactor(forOp);
}

static bool isMuliOrAddi(Operation *op) {
  return isa<MulIOp>(op) || isa<AddIOp>(op);
}

// Ops a slice-reduction loop may accumulate with. Max/Min join add/mul
// because the cross-bank reduce tree only needs associativity and
// commutativity, which they have -- a KNN top-1 is a max reduction and folds
// across banks exactly like a dot product's sum does. Widening this predicate
// cannot disturb any existing kernel: no lowering emits bits.max or bits.min
// today (verified by grepping the lowered IR of every benchmark).
static bool isReduceCombinerOp(Operation *op) {
  return isMuliOrAddi(op) || isa<MaxOp>(op) || isa<MinOp>(op);
}

// Body ops the flattener knows how to turn into trace records. ReLU joins
// MULI/ADDI as a third linear (non-reduction) compute op.
static bool isReluOp(Operation *op) {
  return isa<AndOp>(op) && op->hasAttr("bits.masked_zero");
}

static bool isFlattenableComputeOp(Operation *op) {
  return isMuliOrAddi(op) || isReluOp(op) || isa<AddIFullOp>(op) ||
         isa<XNOrOp>(op) || isa<RangeScanOp>(op) || isa<MaxOp>(op) ||
         isa<MinOp>(op) || isa<SubIOp>(op);
}

static SmallVector<int64_t> getReduceBanksFromAttr(scf::ForOp forOp) {
  SmallVector<int64_t> banks;
  auto treeAttr = forOp->getAttrOfType<ArrayAttr>("bits.reduce_tree");
  if (!treeAttr || treeAttr.empty())
    return banks;
  // Union over ALL levels: in the channel-aware tree the first level does
  // not necessarily touch every active bank (a channel holding a single
  // bank only joins at the cross-channel phase).
  llvm::DenseSet<int64_t> uniqueBanks;
  for (Attribute levelAttr : treeAttr) {
    auto level = dyn_cast<ArrayAttr>(levelAttr);
    if (!level)
      continue;
    for (Attribute pairAttr : level) {
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
  uint32_t record_size;   // = sizeof(TraceRecord) = 48
  uint64_t num_records;
  int64_t  last_end_time;
};
static_assert(sizeof(TraceHeader) == 32, "TraceHeader size mismatch");

// Record: 48 bytes, one entry per merged event (MULI/ADDI) or per edge (ROWCOPY).
// MULI/ADDI banks are encoded as a 128-bit bitmask in `banks` (covers
// kNumBanks up to 128): bit b lives in banks[b>>6] at position (b&63).
// ROWCOPY emits one record per (src,dst) edge; `banks` is unused (0).
struct TraceRecord {
  int64_t  start;      // offset  0,  8B: cycle start
  int64_t  end;        // offset  8,  8B: cycle end
  uint64_t banks[2];   // offset 16, 16B: 128-bit MULI/ADDI bank bitmask; 0 for ROWCOPY
  uint16_t lhs_bw;     // offset 32,  2B: MULI/ADDI→lhsBitwidth, ROWCOPY→bitwidth
  uint16_t rhs_bw;     // offset 34,  2B: MULI/ADDI→rhsBitwidth, ROWCOPY→0
  uint8_t  kind;       // offset 36,  1B: 0=MULI, 1=ADDI, 2=ROWCOPY, 3=RELU,
                       //                 4=ADDI_WIDE (N+N -> N+1)
  uint8_t  src;        // offset 37,  1B: ROWCOPY→srcBank (0..127), others→0
  uint8_t  dst;        // offset 38,  1B: ROWCOPY→dstBank (0..127), others→0
  uint8_t  pad[9];     // offset 39,  9B: explicit pad to 48 bytes
};
static_assert(sizeof(TraceRecord) == 48, "TraceRecord size mismatch");

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

    if (evt.kind == FlattenedEvent::Kind::Subi) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.subi.lhsBitwidth, evt.subi.rhsBitwidth,
                                 evt.subi.start, evt.subi.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.subi.banks.append(evt.subi.banks.begin(),
                                   evt.subi.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::AddiWide) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.addiWide.lhsBitwidth,
                                 evt.addiWide.rhsBitwidth,
                                 evt.addiWide.start, evt.addiWide.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.addiWide.banks.append(evt.addiWide.banks.begin(),
                                       evt.addiWide.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::Min ||
        evt.kind == FlattenedEvent::Kind::Max) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.minMax.lhsBitwidth, evt.minMax.rhsBitwidth,
                                 evt.minMax.start, evt.minMax.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.minMax.banks.append(evt.minMax.banks.begin(),
                                     evt.minMax.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::RangeScan) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.rangeScan.lhsBitwidth,
                                 evt.rangeScan.rhsBitwidth,
                                 evt.rangeScan.start, evt.rangeScan.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.rangeScan.banks.append(evt.rangeScan.banks.begin(),
                                        evt.rangeScan.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::Xnor) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.xnorOp.lhsBitwidth, evt.xnorOp.rhsBitwidth,
                                 evt.xnorOp.start, evt.xnorOp.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.xnorOp.banks.append(evt.xnorOp.banks.begin(),
                                     evt.xnorOp.banks.end());
      }
      return;
    }

    if (evt.kind == FlattenedEvent::Kind::Relu) {
      auto key = std::make_tuple(static_cast<int64_t>(evt.kind),
                                 evt.relu.lhsBitwidth, evt.relu.rhsBitwidth,
                                 evt.relu.start, evt.relu.end);
      auto [it, inserted] = eventIndex.try_emplace(key, events.size());
      if (inserted) {
        events.push_back(evt);
      } else {
        auto &existing = events[it->second];
        existing.relu.banks.append(evt.relu.banks.begin(), evt.relu.banks.end());
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

  // Sibling loops are chained rather than all starting at t=0. Each scf.for
  // used to get its own schedule beginning at zero, so a second kernel in the
  // same function (e.g. a ReLU consuming a conv's output) overlapped -- in
  // fact preceded -- the loop that produces its input. Advancing a
  // function-level cursor makes a producer/consumer chain dependency-correct.
  // It is deliberately conservative: two genuinely independent loops are
  // serialised rather than overlapped, which can only over-report our own
  // runtime, never under-report it. Functions with a single emitting loop
  // (every GEMV and GEMM benchmark -- a GEMM's outer loop emits nothing
  // because its body holds no muli/addi) are unaffected.
  int64_t funcCursor = 0;
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
    // Intra-bank SIMD packing: P independent iterations share one bank row and
    // execute as a single full-row wave, so each bank runs ceil(itersPerBank/P)
    // waves and then folds the P packed lanes (ceil(log2 P) levels).
    int64_t packFactor = getLoopPackFactor(forOp);
    int64_t wavesPerBank = (itersPerBank + packFactor - 1) / packFactor;
    int64_t foldLevels =
        packFactor > 1 ? (int64_t)llvm::Log2_64_Ceil(packFactor) : 0;

    Operation *reduceProducer = nullptr;
    if (auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
      if (yield.getNumOperands() == 1)
        reduceProducer = yield.getOperand(0).getDefiningOp();
    }

    SmallVector<Operation *> linearOps;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isFlattenableComputeOp(&op))
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
    if (reduceProducer && isReduceCombinerOp(reduceProducer)) {
      reduceLatency = getComputeLatency(reduceProducer);
      if (reduceProducer->getNumOperands() >= 1)
        reduceBitwidth = getSliceBitwidth(reduceProducer->getOperand(0));
    }
    // Lossless widening starts from the addends' own width, not the
    // accumulator's declared one; see reduceAddendWidth.
    int64_t reduceBase =
        kWideAdd ? reduceAddendWidth(reduceProducer, reduceBitwidth)
                 : reduceBitwidth;

    for (int64_t bank : activeBanks) {
      int64_t cursor = bankReady[bank];
      for (int64_t wave = 0; wave < wavesPerBank; ++wave) {
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
            localEvents.push_back(makeAddiEvent(add, start, end, bank));
          } else if (auto sub = dyn_cast<SubIOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Subi;
            evt.start = start;
            evt.end = end;
            evt.subi.lhsBitwidth = getSliceBitwidth(sub.getLhs());
            evt.subi.rhsBitwidth = getSliceBitwidth(sub.getRhs());
            evt.subi.start = start;
            evt.subi.end = end;
            evt.subi.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (auto addFull = dyn_cast<AddIFullOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::AddiWide;
            evt.start = start;
            evt.end = end;
            evt.addiWide.lhsBitwidth = getSliceBitwidth(addFull.getLhs());
            evt.addiWide.rhsBitwidth = getSliceBitwidth(addFull.getRhs());
            evt.addiWide.start = start;
            evt.addiWide.end = end;
            evt.addiWide.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (isa<MaxOp>(op) || isa<MinOp>(op)) {
            FlattenedEvent evt;
            evt.kind = isa<MaxOp>(op) ? FlattenedEvent::Kind::Max
                                      : FlattenedEvent::Kind::Min;
            evt.start = start;
            evt.end = end;
            evt.minMax.lhsBitwidth = getSliceBitwidth(op->getOperand(0));
            evt.minMax.rhsBitwidth = getSliceBitwidth(op->getOperand(1));
            evt.minMax.start = start;
            evt.minMax.end = end;
            evt.minMax.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (auto scan = dyn_cast<RangeScanOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::RangeScan;
            evt.start = start;
            evt.end = end;
            evt.rangeScan.lhsBitwidth = getSliceBitwidth(scan.getInput());
            evt.rangeScan.rhsBitwidth = 0;
            evt.rangeScan.start = start;
            evt.rangeScan.end = end;
            evt.rangeScan.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (auto xnorOp = dyn_cast<XNOrOp>(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Xnor;
            evt.start = start;
            evt.end = end;
            evt.xnorOp.lhsBitwidth = getSliceBitwidth(xnorOp.getLhs());
            evt.xnorOp.rhsBitwidth = getSliceBitwidth(xnorOp.getRhs());
            evt.xnorOp.start = start;
            evt.xnorOp.end = end;
            evt.xnorOp.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (isReluOp(op)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Relu;
            evt.start = start;
            evt.end = end;
            evt.relu.lhsBitwidth = getSliceBitwidth(op->getResult(0));
            evt.relu.rhsBitwidth = 0;
            evt.relu.start = start;
            evt.relu.end = end;
            evt.relu.banks.push_back(bank);
            localEvents.push_back(evt);
          }
          cursor = end;
        }
        if (reduceProducer && wave > 0) {
          // Wave `wave` folds its products into a partial sum that now holds
          // `wave + 1` of them, so the lossless width is taken at that count.
          int64_t wIn = losslessReduceWidth(reduceBase, wave);
          int64_t wOut = losslessReduceWidth(reduceBase, wave + 1);
          bool grows = wOut > wIn;
          int64_t start = cursor;
          int64_t end = start + (kWideAdd && isa<AddIOp>(reduceProducer)
                                     ? reduceAddLatencyForWidth(wIn, grows)
                                     : reduceLatency);
          if (auto add = dyn_cast<AddIOp>(reduceProducer)) {
            localEvents.push_back(
                makeReduceAddEvent(add, start, end, bank, wIn, grows));
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
          } else if (isa<MaxOp>(reduceProducer) || isa<MinOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = isa<MaxOp>(reduceProducer) ? FlattenedEvent::Kind::Max
                                                  : FlattenedEvent::Kind::Min;
            evt.start = start;
            evt.end = end;
            evt.minMax.lhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(0));
            evt.minMax.rhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(1));
            evt.minMax.start = start;
            evt.minMax.end = end;
            evt.minMax.banks.push_back(bank);
            localEvents.push_back(evt);
          }
          cursor = end;
        }
      }
      // Intra-bank fold of the P packed lanes into one partial sum before the
      // inter-bank tree. Each level is a cost-proxy intra-bank row copy
      // (src == dst bank) plus one reduce op (decision: reuse cross-bank cost).
      if (reduceProducer && isReduceCombinerOp(reduceProducer)) {
        for (int64_t level = 0; level < foldLevels; ++level) {
          // Folding the packed lanes doubles the item count each level; the
          // copy that stages the other half moves a partial sum, so it is as
          // wide as the sum is.
          int64_t foldItems = wavesPerBank << level;
          int64_t wIn = losslessReduceWidth(reduceBase, foldItems);
          int64_t wOut = losslessReduceWidth(reduceBase, foldItems * 2);
          bool grows = wOut > wIn;
          int64_t copyStart = cursor;
          int64_t copyEnd =
              copyStart + kLaneFoldCostPerBit * costWidth(std::max<int64_t>(1, wIn));
          FlattenedEvent copyEvt;
          copyEvt.kind = FlattenedEvent::Kind::RowCopy;
          copyEvt.start = copyStart;
          copyEvt.end = copyEnd;
          copyEvt.rowCopy.bitwidth = wIn;
          copyEvt.rowCopy.start = copyStart;
          copyEvt.rowCopy.end = copyEnd;
          copyEvt.rowCopy.edges.push_back({bank, bank});
          localEvents.push_back(copyEvt);
          int64_t addStart = copyEnd;
          int64_t addEnd = addStart + (kWideAdd && isa<AddIOp>(reduceProducer)
                                           ? reduceAddLatencyForWidth(wIn, grows)
                                           : reduceLatency);
          if (auto add = dyn_cast<AddIOp>(reduceProducer)) {
            localEvents.push_back(
                makeReduceAddEvent(add, addStart, addEnd, bank, wIn, grows));
          } else if (auto mul = dyn_cast<MulIOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = FlattenedEvent::Kind::Muli;
            evt.start = addStart;
            evt.end = addEnd;
            evt.muli.lhsBitwidth = getSliceBitwidth(mul.getLhs());
            evt.muli.rhsBitwidth = getSliceBitwidth(mul.getRhs());
            evt.muli.start = addStart;
            evt.muli.end = addEnd;
            evt.muli.banks.push_back(bank);
            localEvents.push_back(evt);
          } else if (isa<MaxOp>(reduceProducer) || isa<MinOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = isa<MaxOp>(reduceProducer) ? FlattenedEvent::Kind::Max
                                                  : FlattenedEvent::Kind::Min;
            evt.start = addStart;
            evt.end = addEnd;
            evt.minMax.lhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(0));
            evt.minMax.rhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(1));
            evt.minMax.start = addStart;
            evt.minMax.end = addEnd;
            evt.minMax.banks.push_back(bank);
            localEvents.push_back(evt);
          }
          cursor = addEnd;
        }
      }
      bankReady[bank] = cursor;
    }

    auto treeAttr = forOp->getAttrOfType<ArrayAttr>("bits.reduce_tree");
    if (treeAttr && reduceProducer) {
      // Channel data-bus cursors: copies whose src or dst share a channel
      // serialise on that channel's bus; distinct channels overlap. Bank
      // readiness still orders copies against the producing compute.
      DenseMap<int64_t, int64_t> chanReady;
      // Item count entering the tree: every bank already holds the sum of its
      // own wavesPerBank * packFactor products, and each tree level doubles it.
      int64_t treeItems = wavesPerBank * std::max<int64_t>(1, packFactor);
      for (Attribute levelAttr : treeAttr) {
        auto level = dyn_cast<ArrayAttr>(levelAttr);
        if (!level)
          continue;
        int64_t wIn = losslessReduceWidth(reduceBase, treeItems);
        int64_t wOut = losslessReduceWidth(reduceBase, treeItems * 2);
        bool grows = wOut > wIn;
        treeItems *= 2;
        // Phase A: schedule the level's copies. Copies sharing a channel
        // serialise on that channel's bus; distinct channels overlap.
        SmallVector<int64_t> levelDstBanks;
        int64_t levelEnd = 0;
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
          int64_t srcChan = channelOfBank(srcBank);
          int64_t dstChan = channelOfBank(dstBank);
          int64_t copyCost = copyCostPerBit(srcBank, dstBank) *
                             costWidth(std::max<int64_t>(1, wIn));
          int64_t copyStart =
              std::max({bankReady[srcBank], bankReady[dstBank],
                        chanReady[srcChan], chanReady[dstChan]});
          int64_t copyEnd = copyStart + copyCost;
          chanReady[srcChan] = copyEnd;
          chanReady[dstChan] = copyEnd;
          bankReady[srcBank] = copyEnd;
          bankReady[dstBank] = copyEnd;
          levelEnd = std::max(levelEnd, copyEnd);
          levelDstBanks.push_back(dstBank);
          FlattenedEvent copyEvt;
          copyEvt.kind = FlattenedEvent::Kind::RowCopy;
          copyEvt.start = copyStart;
          copyEvt.end = copyEnd;
          copyEvt.rowCopy.bitwidth = wIn;
          copyEvt.rowCopy.start = copyStart;
          copyEvt.rowCopy.end = copyEnd;
          copyEvt.rowCopy.edges.push_back({srcBank, dstBank});
          localEvents.push_back(copyEvt);
        }
        // Phase B: one SIMD reduce wave per level -- all dst banks run the
        // reduce op concurrently once every copy of the level has landed
        // (shared start/end lets recordEvent merge them into one record).
        int64_t addStart = levelEnd;
        int64_t addEnd = addStart + (kWideAdd && isa<AddIOp>(reduceProducer)
                                         ? reduceAddLatencyForWidth(wIn, grows)
                                         : reduceLatency);
        for (int64_t dstBank : levelDstBanks) {
          if (auto add = dyn_cast<AddIOp>(reduceProducer)) {
            localEvents.push_back(
                makeReduceAddEvent(add, addStart, addEnd, dstBank, wIn, grows));
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
          } else if (isa<MaxOp>(reduceProducer) || isa<MinOp>(reduceProducer)) {
            FlattenedEvent evt;
            evt.kind = isa<MaxOp>(reduceProducer) ? FlattenedEvent::Kind::Max
                                                  : FlattenedEvent::Kind::Min;
            evt.start = addStart;
            evt.end = addEnd;
            evt.minMax.lhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(0));
            evt.minMax.rhsBitwidth =
                getSliceBitwidth(reduceProducer->getOperand(1));
            evt.minMax.start = addStart;
            evt.minMax.end = addEnd;
            evt.minMax.banks.push_back(dstBank);
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
      int64_t offset = funcCursor + wave * localLatency;
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
          } else if (shifted.kind == FlattenedEvent::Kind::Subi) {
            shifted.subi.start = shifted.start;
            shifted.subi.end = shifted.end;
            shifted.subi.banks[0] =
                (shifted.subi.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::AddiWide) {
            shifted.addiWide.start = shifted.start;
            shifted.addiWide.end = shifted.end;
            shifted.addiWide.banks[0] =
                (shifted.addiWide.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::Relu) {
            shifted.relu.start = shifted.start;
            shifted.relu.end = shifted.end;
            shifted.relu.banks[0] = (shifted.relu.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::Min ||
                     shifted.kind == FlattenedEvent::Kind::Max) {
            shifted.minMax.start = shifted.start;
            shifted.minMax.end = shifted.end;
            shifted.minMax.banks[0] =
                (shifted.minMax.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::RangeScan) {
            shifted.rangeScan.start = shifted.start;
            shifted.rangeScan.end = shifted.end;
            shifted.rangeScan.banks[0] =
                (shifted.rangeScan.banks[0] + bankOffset) % kNumBanks;
          } else if (shifted.kind == FlattenedEvent::Kind::Xnor) {
            shifted.xnorOp.start = shifted.start;
            shifted.xnorOp.end = shifted.end;
            shifted.xnorOp.banks[0] =
                (shifted.xnorOp.banks[0] + bankOffset) % kNumBanks;
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
    funcCursor += repeatCount * localLatency;
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
      "/home/tianruiz/papers/cimdram/traces/trace.bin";
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
          r.banks[bank >> 6] |= (1ull << (bank & 63));  // 128-bit bank bitmask
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Addi:
        r.kind   = 1;
        r.lhs_bw = static_cast<uint16_t>(event.addi.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.addi.rhsBitwidth);
        for (int64_t bank : event.addi.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));  // 128-bit bank bitmask
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Subi:
        r.kind   = 9;
        r.lhs_bw = static_cast<uint16_t>(event.subi.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.subi.rhsBitwidth);
        for (int64_t bank : event.subi.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::AddiWide:
        r.kind   = 4;
        r.lhs_bw = static_cast<uint16_t>(event.addiWide.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.addiWide.rhsBitwidth);
        for (int64_t bank : event.addiWide.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Relu:
        r.kind   = 3;
        r.lhs_bw = static_cast<uint16_t>(event.relu.lhsBitwidth);
        r.rhs_bw = 0;
        for (int64_t bank : event.relu.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Min:
      case FlattenedEvent::Kind::Max:
        r.kind   = event.kind == FlattenedEvent::Kind::Min ? 7 : 8;
        r.lhs_bw = static_cast<uint16_t>(event.minMax.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.minMax.rhsBitwidth);
        for (int64_t bank : event.minMax.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::RangeScan:
        r.kind   = 6;
        r.lhs_bw = static_cast<uint16_t>(event.rangeScan.lhsBitwidth);
        r.rhs_bw = 0;
        for (int64_t bank : event.rangeScan.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
        traceEmit(r);
        break;
      case FlattenedEvent::Kind::Xnor:
        r.kind   = 5;
        r.lhs_bw = static_cast<uint16_t>(event.xnorOp.lhsBitwidth);
        r.rhs_bw = static_cast<uint16_t>(event.xnorOp.rhsBitwidth);
        for (int64_t bank : event.xnorOp.banks)
          r.banks[bank >> 6] |= (1ull << (bank & 63));
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

// Intra-bank SIMD packing factor for a leaf slice-reduction loop: how many
// independent iterations fit side-by-side in one bank row, i.e.
// floor(kBankRowWidth / V) where V is the widest body op vector length. The
// P packed iterations compute concurrently as one full-row wave; their P
// partial sums are then folded within the bank (ceil(log2 P) levels). Returns
// 1 (no packing) for any non-leaf or non-reduction loop. Scope: leaf
// slice-reduction loops only.
static int64_t getLoopPackFactor(scf::ForOp forOp) {
  if (!isSliceReductionFor(forOp) || hasNestedForOp(forOp))
    return 1;
  int64_t maxVecLen = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isReduceCombinerOp(&op) || op.getNumResults() == 0)
      continue;
    maxVecLen = std::max(maxVecLen, getSliceVectorLength(op.getResult(0)));
  }
  if (maxVecLen <= 0)
    return 1;
  return std::max<int64_t>(1, kBankRowWidth / maxVecLen);
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
  int64_t reduceBase =
      kWideAdd ? reduceAddendWidth(reduceProducer, reduceBitwidth)
               : reduceBitwidth;

  int64_t packFactor = getCostPackFactor(forOp);
  int64_t foldLevels =
      packFactor > 1 ? (int64_t)llvm::Log2_64_Ceil(packFactor) : 0;
  bool isAddReduce = reduceProducer && isa<AddIOp>(reduceProducer);

  int64_t best = std::numeric_limits<int64_t>::max();
  for (int64_t banks = 1; banks <= std::min<int64_t>(tripCount, kNumBanks);
       banks <<= 1) {
    int64_t iterPerBank = (tripCount + banks - 1) / banks;
    // Each bank runs ceil(iterPerBank / P) packed waves of pre-reduce work,
    // then (waves - 1) intra-bank reductions across waves, then a ceil(log2 P)
    // intra-bank fold of the P packed lanes, then inter-bank tree reduction.
    int64_t wavesPerBank = (iterPerBank + packFactor - 1) / packFactor;
    int64_t preReduceLatency = bodyLatency - reduceLatency;
    int64_t local = wavesPerBank * preReduceLatency
                  + losslessChainCost(reduceBase, wavesPerBank, isAddReduce,
                                      reduceLatency);
    int64_t foldCost = losslessFoldCost(reduceBase, wavesPerBank, foldLevels,
                                        isAddReduce, reduceLatency);
    int64_t reduceCost =
        estimateReduceTreeCost(banks, reduceBase, reduceLatency,
                               wavesPerBank * packFactor, isAddReduce);
    best = std::min(best, local + foldCost + reduceCost);
  }
  return best == std::numeric_limits<int64_t>::max() ? 0 : best;
}

// Number of iterations of this loop's enclosing PARALLEL loops, i.e. how many
// independent sibling instances of `forOp` exist and can run at the same time.
// The emitter already replicates a loop's events across
// `numBanks / bankGroupStride` disjoint bank groups (see
// `getNearestAncestorConcurrentGroups`), so a leaf reduction that claims only
// P of the numBanks banks does NOT leave numBanks - P banks idle: it leaves
// them to numBanks/P sibling instances of itself. Walk stops at the first
// slice-reduction ancestor, whose iterations are serially dependent and
// therefore cannot overlap.
static int64_t getAncestorParallelIters(scf::ForOp forOp) {
  int64_t iters = 1;
  for (Operation *parent = forOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parentFor = dyn_cast<scf::ForOp>(parent);
    if (!parentFor)
      continue;
    if (isSliceReductionFor(parentFor))
      break;
    auto tripMaybe = getConstTripCount(parentFor);
    if (failed(tripMaybe) || *tripMaybe <= 0)
      break;
    iters = std::min<int64_t>(kNumBanks, iters * *tripMaybe);
    if (iters >= kNumBanks)
      break;
  }
  return std::max<int64_t>(1, iters);
}

// How many sibling instances of this loop share the device when the loop itself
// occupies `banks` of the `availBanks` it was offered. Used to divide the
// per-bank LOCAL work in the bank-count argmins: without it, halving `banks`
// looked like twice as much serial work per bank, so both argmins
// systematically preferred to parallelise the REDUCTION dimension (which has to
// pay for a reduce tree) over the free parallel one.
//
// Deliberately applied to the local term only, not to the reduce tree: the
// concurrent groups' compute really does overlap (distinct banks activate in
// parallel), but their tree copies all contend for the same per-channel data
// bus, so tree cost is not discounted by concurrency.
static int64_t ancestorConcurrencyCredit(int64_t banks, int64_t availBanks,
                                         int64_t ancestorParallelIters) {
  int64_t groups =
      std::max<int64_t>(1, availBanks / std::max<int64_t>(1, banks));
  return std::max<int64_t>(1, std::min(groups, ancestorParallelIters));
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

  // Intra-bank SIMD packing must be modelled here exactly as
  // planLeafForRegions() models it. These two functions pick a bank count for
  // the SAME loop and the emitter uses BOTH: planLeafForRegions decides the
  // reduce tree (hence the active bank set), while this function decides
  // `bankGroupStride` / the ancestor wave-repeat count. When they disagree the
  // emitter reduces over one bank count while charging wave repetitions for
  // another -- e.g. a loop whose leaf plan uses 2 banks was still charged as if
  // all 32/64/128 banks were occupied, making the makespan independent of the
  // bank count. packFactor is 1 for every non-leaf or non-reduction loop, so
  // this only changes leaf slice-reduction loops, which is where the two
  // formulas diverged.
  int64_t packFactor = getCostPackFactor(forOp);
  int64_t foldLevels =
      packFactor > 1 ? (int64_t)llvm::Log2_64_Ceil(packFactor) : 0;
  bool isAddReduce = reduceProducer && isa<AddIOp>(reduceProducer);
  int64_t reduceBase =
      kWideAdd ? reduceAddendWidth(reduceProducer, reduceBitwidth)
               : reduceBitwidth;

  int64_t ancestorParallelIters = getAncestorParallelIters(forOp);

  int64_t bestBanks = 1;
  int64_t bestCost = std::numeric_limits<int64_t>::max();
  for (int64_t banks = 1; banks <= std::min<int64_t>(tripCount, kNumBanks);
       banks <<= 1) {
    int64_t iterPerBank = (tripCount + banks - 1) / banks;
    int64_t wavesPerBank = (iterPerBank + packFactor - 1) / packFactor;
    int64_t local;
    if (isSliceReductionFor(forOp) && reduceLatency > 0) {
      // Same model as estimateForLatency/planLeafForRegions: n packed
      // pre-reduce waves + (n-1) intra-bank reductions + the lane fold. The
      // reduce op is NOT counted in every iteration.
      int64_t preReduceLatency = bodyLatency - reduceLatency;
      local = wavesPerBank * preReduceLatency
            + losslessChainCost(reduceBase, wavesPerBank, isAddReduce,
                                reduceLatency)
            + losslessFoldCost(reduceBase, wavesPerBank, foldLevels,
                               isAddReduce, reduceLatency);
    } else {
      local = wavesPerBank * bodyLatency;
    }
    int64_t reduceCost =
        isSliceReductionFor(forOp)
            ? estimateReduceTreeCost(banks, reduceBase, reduceLatency,
                                     wavesPerBank * packFactor, isAddReduce)
            : 0;
    local /= ancestorConcurrencyCredit(banks, kNumBanks, ancestorParallelIters);
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

// Channel-blind inter-bank reduction tree: plain stride halving over the flat
// bank list, pairing i with i + half at every level. This is what a mapper
// whose cost model cannot tell an intra-channel copy from a cross-channel one
// builds -- with every pairing priced the same there is no reason to prefer
// any particular one, so the simplest tree wins. It is therefore the tree the
// Proxy cost model gets, and the extra cross-channel traffic it creates
// ((N - banksPerChannel) rows instead of (#channels - 1)) is a mapping
// decision the simulator then has to pay for, not a modelling artefact.
static SmallVector<SmallVector<std::pair<int64_t, int64_t>>>
buildStrideHalvingReduceTree(ArrayRef<int64_t> activeBanks) {
  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> levels;
  SmallVector<int64_t> live(activeBanks.begin(), activeBanks.end());
  while (live.size() > 1) {
    size_t n = live.size();
    size_t half = (n + 1) / 2;
    SmallVector<std::pair<int64_t, int64_t>> level;
    for (size_t i = 0; i + half < n; ++i)
      level.push_back({live[i + half], live[i]}); // {src, dst}
    live.truncate(half);
    levels.push_back(std::move(level));
  }
  return levels;
}

// Channel-aware inter-bank reduction tree over an active bank group.
// Phase 1 halves within each channel (per-channel copies run on that
// channel's own data bus; channels proceed in parallel); phase 2 halves
// across the per-channel survivors (cross-channel copies between disjoint
// channel pairs, parallel). Levels are returned outermost first as
// {src_bank, dst_bank} pairs; the final result lands in activeBanks[0],
// same as the legacy stride-halving tree. Compared to that tree this cuts
// the cross-channel rows from (N - banksPerChannel) to (#channels - 1).
//
// With location pricing switched off the channel-blind tree above is used;
// dispatching here (rather than at the call sites) keeps the three consumers
// -- the closed-form estimator, the bits.reduce_tree annotation and the event
// flattener -- reading one and the same tree.
static SmallVector<SmallVector<std::pair<int64_t, int64_t>>>
buildChannelAwareReduceTree(ArrayRef<int64_t> activeBanks) {
  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> levels;
  if (activeBanks.size() <= 1)
    return levels;

  if (kBlindLocation)
    return buildStrideHalvingReduceTree(activeBanks);

  // Group by channel, preserving group order (first bank stays first).
  llvm::MapVector<int64_t, SmallVector<int64_t>> byChannel;
  for (int64_t b : activeBanks)
    byChannel[channelOfBank(b)].push_back(b);

  // Phase 1: intra-channel halving, all channels in lockstep per level.
  bool more = true;
  while (more) {
    more = false;
    SmallVector<std::pair<int64_t, int64_t>> level;
    for (auto &entry : byChannel) {
      SmallVector<int64_t> &list = entry.second;
      size_t n = list.size();
      if (n <= 1)
        continue;
      size_t half = (n + 1) / 2;
      for (size_t i = 0; i + half < n; ++i)
        level.push_back({list[i + half], list[i]}); // {src, dst}
      list.truncate(half);
      if (list.size() > 1)
        more = true;
    }
    if (!level.empty())
      levels.push_back(std::move(level));
  }

  // Phase 2: cross-channel halving over the per-channel survivors.
  SmallVector<int64_t> reps;
  for (auto &entry : byChannel)
    reps.push_back(entry.second.front());
  while (reps.size() > 1) {
    size_t n = reps.size();
    size_t half = (n + 1) / 2;
    SmallVector<std::pair<int64_t, int64_t>> level;
    for (size_t i = 0; i + half < n; ++i)
      level.push_back({reps[i + half], reps[i]}); // {src, dst}
    reps.truncate(half);
    levels.push_back(std::move(level));
  }
  return levels;
}

// Critical-path cost of the inter-bank reduce tree for `banks` consecutive
// active banks (base 0): within a level, copies serialise on each channel's
// data bus (a copy occupies both its src and dst channel bus for its whole
// duration) while distinct channels run in parallel; one reduce-op wave
// (SIMD across the level's dst banks) follows each level.
static int64_t estimateReduceTreeCost(int64_t banks, int64_t bitwidth,
                                      int64_t reduceLatency,
                                      int64_t itemsAtLeaf, bool isAddReduce) {
  if (banks <= 1)
    return 0;
  auto levels = buildChannelAwareReduceTree(buildBankGroup(0, banks));
  llvm::DenseMap<int64_t, int64_t> chanBusy; // bus busy-until per channel
  int64_t cursor = 0;
  // Item count entering the tree: each bank arrives holding the sum of its own
  // `itemsAtLeaf` values, and every level doubles it. Under wide-add the
  // partial sum widens with that count, so both the copies and the combine of a
  // level are priced at the width the sum has reached -- see
  // reduceCombineLatency, which the emitter uses for the same levels.
  int64_t items = std::max<int64_t>(1, itemsAtLeaf);
  for (const auto &level : levels) {
    int64_t bw = std::max<int64_t>(
        1, costWidth(losslessReduceWidth(bitwidth, items)));
    int64_t combine = reduceCombineLatency(bitwidth, items, items * 2,
                                           isAddReduce, reduceLatency);
    int64_t levelEnd = cursor;
    for (const auto &[src, dst] : level) {
      int64_t sc = channelOfBank(src), dc = channelOfBank(dst);
      int64_t start = std::max({cursor, chanBusy[sc], chanBusy[dc]});
      int64_t end = start + copyCostPerBit(src, dst) * bw;
      chanBusy[sc] = end;
      chanBusy[dc] = end;
      levelEnd = std::max(levelEnd, end);
    }
    cursor = levelEnd + combine; // level barrier: copies then reduce
    items *= 2;
  }
  return cursor;
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

// `ancestorParallelIters` = how many independent sibling instances of `forOp`
// the emitter will run concurrently on disjoint bank groups; 1 disables the
// credit. Callers that already model the ancestor sweep themselves (the joint
// outer-regions search below multiplies by `iterPerRegion`) must pass 1, or the
// same concurrency would be counted twice.
static RegionPlan planLeafForRegions(scf::ForOp forOp, int64_t baseBank,
                                     int64_t availBanks,
                                     int64_t ancestorParallelIters = 1) {
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
  int64_t packFactor = getCostPackFactor(forOp);
  int64_t foldLevels =
      packFactor > 1 ? (int64_t)llvm::Log2_64_Ceil(packFactor) : 0;
  bool isAddReduce = reduceProducer && isa<AddIOp>(reduceProducer);
  int64_t reduceBase =
      kWideAdd ? reduceAddendWidth(reduceProducer, reduceBitwidth)
               : reduceBitwidth;

  for (int64_t regions : candidates) {
    int64_t banksPerRegion = std::max<int64_t>(1, availBanks / regions);
    int64_t iterPerRegion = std::max<int64_t>(1, tripCount / regions);
    // For reduction loops each bank runs ceil(iterPerRegion / P) packed waves
    // of pre-reduce work, folds them with (waves - 1) intra-bank reductions
    // plus a ceil(log2 P) lane fold, then joins the inter-bank tree.
    bool isReduce = isSliceReductionFor(forOp) && reduceLatency > 0;
    int64_t wavesPerRegion = (iterPerRegion + packFactor - 1) / packFactor;
    int64_t preReduceLatency = isReduce ? (bodyLatency - reduceLatency) : bodyLatency;
    int64_t local = wavesPerRegion * preReduceLatency;
    int64_t intraReduce =
        isReduce ? losslessChainCost(reduceBase, wavesPerRegion, isAddReduce,
                                     reduceLatency) +
                       losslessFoldCost(reduceBase, wavesPerRegion,
                                        foldLevels, isAddReduce, reduceLatency)
                 : 0;
    int64_t interRegionReduce = 0;
    if (isSliceReductionFor(forOp) && regions > 1)
      interRegionReduce = estimateReduceTreeCost(
          regions, reduceBase, reduceLatency, wavesPerRegion * packFactor,
          isAddReduce);
    // Banks this plan does not claim are filled by concurrent ancestor
    // iterations, so the per-bank local+intra work is shared `credit` ways.
    // The inter-region tree is NOT discounted (channel bus contention).
    int64_t credit =
        ancestorConcurrencyCredit(regions, availBanks, ancestorParallelIters);
    int64_t makespan = (local + intraReduce) / credit + interRegionReduce;
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
    GRBLinExpr cost; // location-aware transfer window (0 when same bank)
  };
  llvm::SmallVector<CloneEdge> cloneEdges;

  // Bitwidth of the value a dataflow edge out of node i transfers (a bw-bit
  // slice moves as bw bit-plane rows); 1 for non-slice results.
  auto xferBitwidth = [&](int i) -> int64_t {
    int64_t bw = 0;
    for (Value r : dag.nodes[i].op->getResults())
      bw = std::max(bw, getSliceBitwidth(r));
    return std::max<int64_t>(1, costWidth(bw));
  };
  const int64_t numChannels =
      (kNumBanks + kBanksPerChannel - 1) / kBanksPerChannel;

  const int64_t M = std::max<int64_t>(100, totalDuration + 1024);
  for (int i = 0; i < N; ++i) {
    for (int j : dag.nodes[i].succs) {
      GRBVar cloneVar =
          model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                       "clone_" + std::to_string(i) + "_" + std::to_string(j));
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

      // Location-aware transfer cost, consistent with copyCostPerBit():
      // 0 when i and j share a bank; kCopyCostIntraChPerBit*bw when they
      // share a channel but not a bank; kCopyCostCrossChPerBit*bw across
      // channels. sameChan (sumBC) is linearised per channel from the
      // bank-assignment vars; same bank implies same channel, so
      // (sumBC - sumB) flags exactly the intra-channel inter-bank case.
      GRBLinExpr sumBC;
      for (int64_t c = 0; c < numChannels; ++c) {
        GRBLinExpr yi, yj;
        for (int64_t k = c * kBanksPerChannel;
             k < std::min<int64_t>((c + 1) * kBanksPerChannel, kNumBanks);
             ++k) {
          yi += x[i][k];
          yj += x[j][k];
        }
        GRBVar bc = model.addVar(
            0.0, 1.0, 0.0, GRB_BINARY,
            "bc_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(c));
        model.addConstr(bc <= yi);
        model.addConstr(bc <= yj);
        model.addConstr(bc >= yi + yj - 1);
        sumBC += bc;
      }
      int64_t bwXfer = xferBitwidth(i);
      GRBLinExpr xferCost =
          kCopyCostIntraChPerBit * bwXfer * (sumBC - sumB) +
          kCopyCostCrossChPerBit * bwXfer * (1 - sumBC);
      cloneEdges.push_back(CloneEdge{i, j, cloneVar, xferCost});

      model.addConstr(start[j] >= end[i] + xferCost,
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
            start[k] >= end[i] + xferCost -
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
            start[k] >= end[i] + xferCost -
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
      // Binary == 1 iff DAG nodes a and b are mapped to the same bank.
      auto sameBankVar = [&](int a, int b, llvm::StringRef tag) -> GRBVar {
        GRBLinExpr sumB;
        for (int bank = 0; bank < kNumBanks; ++bank) {
          GRBVar bVar = model.addVar(
              0.0, 1.0, 0.0, GRB_BINARY,
              "b_" + std::string(tag) + "_" + std::to_string(e1) + "_" +
                  std::to_string(e2) + "_" + std::to_string(bank));
          model.addConstr(bVar <= x[a][bank]);
          model.addConstr(bVar <= x[b][bank]);
          model.addConstr(bVar >= x[a][bank] + x[b][bank] - 1);
          sumB += bVar;
        }
        GRBVar sameBank = model.addVar(
            0.0, 1.0, 0.0, GRB_BINARY,
            "same_" + std::string(tag) + "_" + std::to_string(e1) + "_" +
                std::to_string(e2));
        model.addConstr(sumB == sameBank, "same_link_" + std::string(tag) +
                                              "_" + std::to_string(e1) + "_" +
                                              std::to_string(e2));
        return sameBank;
      };

      // Serialize the two transfers' location-priced windows when their
      // endpoints share a bank, unless `relax` (== 1) marks them as merged
      // into a single row copy and therefore safe to run simultaneously.
      auto addCloneOrderConstraints = [&](GRBVar sameBank, GRBLinExpr relax,
                                          llvm::StringRef tag) {
        GRBVar orderVar = model.addVar(
            0.0, 1.0, 0.0, GRB_BINARY,
            "order_" + std::string(tag) + "_" + std::to_string(e1) + "_" +
                std::to_string(e2));
        model.addConstr(end[c1.src] + c1.cost <=
                            end[c2.src] + M * (3 - c1.active - c2.active -
                                               sameBank + orderVar + relax),
                        "clone_serial_ab_" + std::string(tag) + "_" +
                            std::to_string(e1) + "_" + std::to_string(e2));
        model.addConstr(end[c2.src] + c2.cost <=
                            end[c1.src] +
                                M * (3 - c1.active - c2.active - sameBank +
                                     (1 - orderVar) + relax),
                        "clone_serial_ba_" + std::string(tag) + "_" +
                            std::to_string(e1) + "_" + std::to_string(e2));
      };

      GRBVar sameSrc = sameBankVar(c1.src, c2.src, "src_src");
      GRBVar sameDst = sameBankVar(c1.dst, c2.dst, "dst_dst");
      // Two transfers sharing both source and destination bank move the same
      // row to the same place: a single row copy carries both results, so they
      // need not be serialized. merge = sameSrc AND sameDst.
      GRBVar merge =
          model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                       "merge_" + std::to_string(e1) + "_" + std::to_string(e2));
      model.addConstr(merge <= sameSrc);
      model.addConstr(merge <= sameDst);
      model.addConstr(merge >= sameSrc + sameDst - 1);

      GRBVar sameSrcDst = sameBankVar(c1.src, c2.dst, "src_dst");
      GRBVar sameDstSrc = sameBankVar(c1.dst, c2.src, "dst_src");

      // Same src+dst (merge) relaxes the src/src and dst/dst windows. A bank
      // acting as both a source and a destination (src/dst, dst/src) is a real
      // conflict and is always serialized.
      addCloneOrderConstraints(sameSrc, merge, "src_src");
      addCloneOrderConstraints(sameDst, merge, "dst_dst");
      addCloneOrderConstraints(sameSrcDst, GRBLinExpr(0.0), "src_dst");
      addCloneOrderConstraints(sameDstSrc, GRBLinExpr(0.0), "dst_src");
    }
  }

  // Intra-bank SIMD packing. Independent, non-container ops of the same kind
  // and bitwidth signature may share a bank and run as one concurrent wave so
  // long as the total vector length of all ops co-located in that bank's row
  // stays within kBankRowWidth. For every packable op we accumulate the vector
  // lengths of its co-packed partners into rowCols[i] and cap the sum below.
  std::vector<bool> packable(N, false);
  std::vector<int64_t> vlen(N, 0);
  std::vector<GRBLinExpr> rowCols(N);
  for (int i = 0; i < N; ++i) {
    if (dag.nodes[i].isContainer)
      continue;
    if (!getPackSignature(dag.nodes[i].op).packable())
      continue;
    int64_t v = getSliceVectorLength(dag.nodes[i].op->getResult(0));
    if (v <= 0 || v > kBankRowWidth)
      continue; // a single op wider than a row cannot pack; keep it serial.
    packable[i] = true;
    vlen[i] = v;
    rowCols[i] = (double)v;
  }

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      if (dep[i][j] || dep[j][i] || dag.nodes[i].isContainer ||
          dag.nodes[j].isContainer)
        continue;
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

      bool canPack = packable[i] && packable[j] &&
                     getPackSignature(dag.nodes[i].op) ==
                         getPackSignature(dag.nodes[j].op);
      if (!canPack) {
        // Incompatible (different kind/bitwidth, or non-packable): the two ops
        // must run serially whenever they share a bank.
        model.addConstr(start[j] >= end[i] - M * (diffVar + 1.0 - orderVar),
                        "serial_ij_" + std::to_string(i) + "_" +
                            std::to_string(j));
        model.addConstr(start[i] >= end[j] - M * (diffVar + orderVar),
                        "serial_ji_" + std::to_string(i) + "_" +
                            std::to_string(j));
        continue;
      }

      // Packable pair: pack == 1 places them in the same bank at the same start
      // time (equal latency => equal end time => fully simultaneous); pack == 0
      // falls back to serialization. Row capacity is enforced via rowCols.
      GRBVar pack =
          model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                       "pack_" + std::to_string(i) + "_" + std::to_string(j));
      model.addConstr(pack + diffVar <= 1, "pack_samebank_" +
                                               std::to_string(i) + "_" +
                                               std::to_string(j));
      model.addConstr(start[i] - start[j] <= M * (1 - pack),
                      "pack_start_lo_" + std::to_string(i) + "_" +
                          std::to_string(j));
      model.addConstr(start[j] - start[i] <= M * (1 - pack),
                      "pack_start_hi_" + std::to_string(i) + "_" +
                          std::to_string(j));
      model.addConstr(start[j] >=
                          end[i] - M * (diffVar + pack + 1.0 - orderVar),
                      "serial_ij_" + std::to_string(i) + "_" +
                          std::to_string(j));
      model.addConstr(start[i] >= end[j] - M * (diffVar + pack + orderVar),
                      "serial_ji_" + std::to_string(i) + "_" +
                          std::to_string(j));
      rowCols[i] += (double)vlen[j] * pack;
      rowCols[j] += (double)vlen[i] * pack;
    }
  }

  // Bank-row capacity: the vector lengths of all ops packed together in one
  // bank row may not exceed kBankRowWidth.
  for (int i = 0; i < N; ++i) {
    if (packable[i])
      model.addConstr(rowCols[i] <= (double)kBankRowWidth,
                      "row_cap_" + std::to_string(i));
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

    // Apply pass options to the file-scope model parameters.
    if (numBanks < 1 || numBanks > 128 ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(numBanks))) {
      func.emitError("bits-opt-mapping: num-banks must be a power of two in "
                     "[1, 128], got ")
          << numBanks;
      signalPassFailure();
      return;
    }
    if (memType == "ddr") {
      kBanksPerChannel = 32;
      kCopyCostIntraChPerBit = 19;
      kCopyCostCrossChPerBit = 10;
    } else if (memType == "hbm") {
      kBanksPerChannel = 16;
      kCopyCostIntraChPerBit = 6;
      kCopyCostCrossChPerBit = 3;
    } else {
      func.emitError("bits-opt-mapping: mem-type must be 'ddr' or 'hbm', "
                     "got '")
          << memType << "'";
      signalPassFailure();
      return;
    }
    if (costModel == "exact") {
      kCostModel = CostModel::Exact;
      kBlindOpLevel = kBlindWidth = kBlindLocation = kBlindPack = false;
    } else if (costModel == "proxy") {
      kCostModel = CostModel::Proxy;
      // Which modelling features the proxy is missing. Default: all of them.
      StringRef axes(proxyBlind);
      if (axes.trim().empty())
        axes = "width,location,pack";
      SmallVector<StringRef> parts;
      axes.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (StringRef raw : parts) {
        StringRef a = raw.trim();
        if (a == "oplevel")
          kBlindOpLevel = true;
        else if (a == "width")
          kBlindWidth = kBlindOpLevel = true; // no widths at all implies no
                                              // per-op counts either
        else if (a == "location")
          kBlindLocation = true;
        else if (a == "pack")
          kBlindPack = true;
        else {
          func.emitError("bits-opt-mapping: proxy-blind axis must be one of "
                         "oplevel/width/location/pack, got '")
              << a << "'";
          signalPassFailure();
          return;
        }
      }
      if (kBlindLocation) {
        // A location-blind model charges the same for every copy. Overriding
        // the two globals (rather than special-casing copyCostPerBit) keeps the
        // closed-form estimators and the ILP's linearised transfer cost reading
        // the same numbers -- they diverged once before and it cost a
        // benchmark's worth of wrong schedules.
        kCopyCostIntraChPerBit = kLaneFoldCostPerBit;
        kCopyCostCrossChPerBit = kLaneFoldCostPerBit;
      }
    } else {
      func.emitError("bits-opt-mapping: cost-model must be 'exact' or "
                     "'proxy', got '")
          << costModel << "'";
      signalPassFailure();
      return;
    }
    if (backend == "simdram") {
      kBackend = Backend::Simdram;
    } else if (backend == "prada") {
      kBackend = Backend::Prada;
    } else {
      func.emitError("bits-opt-mapping: backend must be 'simdram' or "
                     "'prada', got '")
          << backend << "'";
      signalPassFailure();
      return;
    }
    kWideAdd = wideAdd;
    kNumBanks = numBanks;
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
          // Lifting a loop body into the ILP costs nodes*banks assignment
          // binaries plus per-edge per-channel indicators, so a body with a
          // long dependency chain explodes the model: a 63-node popcount
          // adder tree at 128 banks produced 6.7M rows / 2.25M columns and
          // did not solve. It is also pure waste -- the trace emitter derives
          // a loop's schedule from the loop's own attributes and its own
          // cursor over the body ops, and never reads bits.start_time or
          // bits.bank_id off a body op. So cap the body size we are willing
          // to lift and otherwise price the loop in aggregate.
          int64_t bodyComputeOps = 0;
          for (Operation &bodyOp : forOp.getBody()->without_terminator())
            if (isSupportedBitsSliceComputeOp(&bodyOp))
              ++bodyComputeOps;
          bool bodyTooLargeForIlp = bodyComputeOps > kMaxIlpNodesPerLoopBody;
          bool aggregateLoopWork = hasSliceReduction || nestedFor ||
                                   spansMultipleBankWaves ||
                                   bodyTooLargeForIlp;
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

      // Annotate intra-bank SIMD packing groups so the chosen mapping is
      // inspectable. Packable ops the ILP placed on the same bank at the same
      // start time with the same pack signature form one concurrent wave; each
      // such wave (size >= 2) receives a distinct bits.simd_group id.
      {
        std::map<std::tuple<int, int64_t, int, int64_t, int64_t>,
                 SmallVector<Operation *>>
            waves;
        for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
          if (chosen[i] < 0 || dag.nodes[i].isContainer)
            continue;
          PackSignature sig = getPackSignature(dag.nodes[i].op);
          if (!sig.packable())
            continue;
          waves[{chosen[i], solution.startTime[i], sig.kind, sig.lhsBitwidth,
                 sig.rhsBitwidth}]
              .push_back(dag.nodes[i].op);
        }
        int64_t groupId = 0;
        for (auto &[key, members] : waves) {
          (void)key;
          if (members.size() < 2)
            continue;
          for (Operation *member : members)
            member->setAttr("bits.simd_group",
                            attrBuilder.getI64IntegerAttr(groupId));
          ++groupId;
        }
      }

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
        // Intra-bank SIMD packing: P iterations per bank row run as one wave,
        // so each bank does ceil(itersPerBank/P) waves with (waves-1) intra-bank
        // reductions, plus a ceil(log2 P) lane fold.
        int64_t packFactor = getLoopPackFactor(lp);
        int64_t wavesPerBank = (itersPerBank + packFactor - 1) / packFactor;
        lp->setAttr("bits.pack_factor",
                    attrBuilder.getI64IntegerAttr(packFactor));
        lp->setAttr("bits.intra_reduce_count",
                    attrBuilder.getI64IntegerAttr(
                        std::max<int64_t>(0, wavesPerBank - 1)));
        if (packFactor > 1)
          lp->setAttr("bits.fold_levels",
                      attrBuilder.getI64IntegerAttr(
                          (int64_t)llvm::Log2_64_Ceil(packFactor)));
        int64_t reduceLevels =
            parallelFactor > 1 ? (int64_t)llvm::Log2_64(parallelFactor) : 0;
        lp->setAttr("bits.reduce_levels",
                    attrBuilder.getI64IntegerAttr(reduceLevels));
        // Build the inter-bank tree descriptor using the ILP-assigned base
        // bank. Channel-aware: halve within each channel first (copies stay
        // on that channel's data bus), then across the per-channel survivors
        // (one cross-channel copy per surviving pair). The final result
        // lands in baseBank, as with the legacy stride-halving tree.
        int64_t baseBank = 0;
        if (auto bankAttr = lp->getAttrOfType<IntegerAttr>("bits.bank_id"))
          baseBank = bankAttr.getInt();
        auto levels = buildChannelAwareReduceTree(
            buildBankGroup(baseBank, parallelFactor));
        SmallVector<Attribute> treeLevels;
        for (const auto &level : levels) {
          SmallVector<Attribute> levelPairs;
          for (const auto &[srcBank, dstBank] : level) {
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
          // `allForOps` comes from a POST-order walk, so childFor was already
          // visited and annotated as a leaf loop by the else-branch below. Its
          // leaf plan is the one the trace emitter actually executes: the
          // emitter derives its active bank set from `bits.reduce_tree`
          // (getReduceBanksFromAttr), never from `bits.parallel_factor`.
          // Re-annotating here used to overwrite parallel_factor/reduce_levels
          // with bestInnerParallelFactor while leaving the leaf plan's
          // reduce_tree in place -- annotateForLoop only writes reduce_tree
          // when the tree is non-empty, so a factor-1 rewrite silently kept a
          // stale N-bank tree. That made the annotations self-contradictory
          // (e.g. parallel_factor=1 alongside a 32-bank tree) and made them
          // disagree with the schedule that was actually emitted.
          //
          // Keep the leaf plan authoritative so the attributes describe the
          // emitted schedule. This is deliberately annotation-only: the emitted
          // trace is unchanged, byte for byte.
          if (!childFor->hasAttr("bits.parallel_factor")) {
            auto childTripMaybe = getConstTripCount(childFor);
            int64_t childTripCount =
                succeeded(childTripMaybe) ? *childTripMaybe : 1;
            annotateForLoop(childFor, bestInnerParallelFactor, childTripCount,
                            isSliceReductionFor(childFor));
          }

        } else {
          RegionPlan plan = planLeafForRegions(
              forOp, 0, kNumBanks, getAncestorParallelIters(forOp));
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
