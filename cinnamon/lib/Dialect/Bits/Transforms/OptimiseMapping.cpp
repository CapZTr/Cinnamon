#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"
#include "gurobi_c++.h"
#include "gurobi_c.h"

#include <exception>
#include <iostream>
#include <algorithm>
#include <limits>
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

constexpr int64_t kNumBanks = 64;
constexpr int64_t kCloneCostPerBit = 3;
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
    return 3 * getSliceBitwidth(add.getResult());
  if (auto mul = dyn_cast<MulIOp>(op))
    return 4 * getSliceBitwidth(mul.getLhs()) * getSliceBitwidth(mul.getRhs()) -
           3 * getSliceBitwidth(mul.getLhs());
  if (auto andOp = dyn_cast<AndOp>(op))
    return getSliceBitwidth(andOp.getResult());
  if (auto orOp = dyn_cast<OrOp>(op))
    return getSliceBitwidth(orOp.getResult());
  if (auto xorOp = dyn_cast<XOrOp>(op))
    return 3 * getSliceBitwidth(xorOp.getResult());

  if (op->getNumResults() == 0)
    return 0;
  return 2 * getSliceBitwidth(op->getResult(0));
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
    // Non-reduction loops (both nested and non-nested) can be fully parallelized
    // across bank groups: each outer iteration runs independently on its own bank
    // group. Use ceil(tripCount / kNumBanks) to model available parallelism.
    return bodyLatency * ((tripCount + kNumBanks - 1) / kNumBanks);
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
    int64_t local = iterPerBank * bodyLatency;
    int64_t reduceTreeLevels = llvm::Log2_64(banks);
    int64_t reduceCost = reduceTreeLevels * (cloneLatency + reduceLatency);
    best = std::min(best, local + reduceCost);
  }
  return best == std::numeric_limits<int64_t>::max() ? 0 : best;
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
    int64_t local = iterPerBank * bodyLatency;
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
    int64_t local = iterPerRegion * bodyLatency;
    int64_t reduceInRegion = 0;
    if (isSliceReductionFor(forOp) && reduceLatency > 0)
      reduceInRegion = std::max<int64_t>(0, iterPerRegion - 1) * reduceLatency;
    int64_t interRegionReduce = 0;
    if (isSliceReductionFor(forOp) && regions > 1)
      interRegionReduce = llvm::Log2_64(regions) * (reduceLatency + cloneLatency);
    int64_t makespan = local + reduceInRegion + interRegionReduce;

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
      // For-loop nodes receive bits.bank_id (the ILP-assigned base bank).
      // Top-level slice compute ops (non-loop nodes in the DAG) additionally
      // receive bits.start_time and, for each result consumed by an op on a
      // different bank, bits.out_xfer = [{dst_bank, time}, ...] where time is
      // the ILP end cycle of this op (earliest the source bank can send).
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

        SmallVector<Attribute> outXfers;
        for (Value result : op->getResults()) {
          for (Operation *user : result.getUsers()) {
            auto it = dag.opToNodeID.find(user);
            if (it == dag.opToNodeID.end())
              continue;
            int j = it->second;
            if (chosen[j] >= 0 && chosen[j] != chosen[i]) {
              NamedAttrList xfer;
              xfer.set("dst_bank",
                       IntegerAttr::get(IntegerType::get(ctx, 64), chosen[j]));
              xfer.set("time",
                       IntegerAttr::get(IntegerType::get(ctx, 64),
                                        solution.endTime[i]));
              outXfers.push_back(DictionaryAttr::get(ctx, xfer.getAttrs()));
            }
          }
        }
        if (!outXfers.empty())
          op->setAttr("bits.out_xfer", ArrayAttr::get(ctx, outXfers));
      }
 
      // Phase 5: for-loop parallelism planning.
      // Each scf.for receives exactly one attribute:
      //
      //   bits.parallel_factor  (i64)
      //     Number of independent execution units this loop is split across:
      //       non-reduction loop   -> number of parallel outer instances
      //       slice-reduction loop -> number of banks computing partial sums
      //
      // The dispatch codegen derives all other scheduling parameters:
      //   iters_per_unit     = tripCount / parallel_factor
      //   banks_per_instance = kNumBanks / outer_parallel_factor  (outer only)
      //   reduce_levels      = log2(parallel_factor)              (reduction only)
      //
      // A two-level nest (non-reduction outer + reduction inner) is planned
      // jointly so the bank budget is split correctly. The child for is
      // annotated inside the outer loop's planning and skipped when the walk
      // reaches it independently.
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
          // Joint optimisation for a two-level nest: find the outer instance
          // count (bestOuterRegions) and inner bank count
          // (bestInnerParallelFactor) minimising
          //   outer_iters_per_instance * inner_makespan.
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

          forOp->setAttr("bits.parallel_factor",
                         attrBuilder.getI64IntegerAttr(bestOuterRegions));
          childFor->setAttr("bits.parallel_factor",
                            attrBuilder.getI64IntegerAttr(bestInnerParallelFactor));
        } else {
          RegionPlan plan = planLeafForRegions(forOp, 0, kNumBanks);
          forOp->setAttr("bits.parallel_factor",
                         attrBuilder.getI64IntegerAttr(plan.regionCount));
        }
      }
    } catch (GRBException &e) {
      std::cerr << "Gurobi error: " << e.getMessage() << "\n";
    } catch (std::exception &ex) {
      std::cerr << "Error: " << ex.what() << "\n";
    }
  }
};

} // namespace mlir::bits
