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

static int64_t estimateLoopBodyLatency(scf::ForOp forOp) {
  int64_t bodyLatency = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(&op)) {
      auto innerTripCount = getConstTripCount(innerFor);
      if (failed(innerTripCount))
        continue;
      bodyLatency += *innerTripCount * estimateLoopBodyLatency(innerFor);
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

  if (!isSliceReductionFor(forOp))
    return bodyLatency * ((tripCount + kNumBanks - 1) / kNumBanks);

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
      best.unrollFactor = iterPerRegion;
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
      if (!dep[i][j] && !dep[j][i]) {
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
      int64_t end = start + std::max<int64_t>(1, dag.nodes[i].duration);
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
    DenseMap<Operation *, bool> loopHasSliceReduction;
    DenseMap<Operation *, int> loopReduceProducerNode;

    auto collectNodes = [&](auto &&self, Region &region, int parentForNode,
                            int64_t loopDepth) -> LogicalResult {
      for (Operation &op : region.getOps()) {
        if (isSupportedBitsSliceComputeOp(&op)) {
          int id = dag.nodes.size();
          dag.nodes.push_back(
              Node{&op, getComputeLatency(&op), parentForNode, loopDepth, {}, {}});
          dag.opToNodeID[&op] = id;
          continue;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          FailureOr<int64_t> tripCount = getConstTripCount(forOp);
          if (failed(tripCount) || !llvm::isPowerOf2_64(*tripCount)) {
            forOp.emitOpError("requires constant power-of-two trip count");
            return failure();
          }
          int64_t estLatency = estimateForLatency(forOp);
          int id = dag.nodes.size();
          dag.nodes.push_back(Node{forOp.getOperation(), estLatency, parentForNode,
                                   loopDepth, {}, {}});
          dag.opToNodeID[&op] = id;
          loopTripCount[&op] = *tripCount;
          loopBestBanks[&op] = estimateBestParallelBanksForLoop(forOp);
          loopHasSliceReduction[&op] = isSliceReductionFor(forOp);
          loopReduceProducerNode[&op] = -1;
          if (failed(self(self, forOp.getRegion(), id, loopDepth + 1)))
            return failure();
          if (auto yield =
                  dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
            if (yield.getNumOperands() == 1) {
              if (Operation *producer = yield.getOperand(0).getDefiningOp()) {
                if (dag.opToNodeID.contains(producer))
                  loopReduceProducerNode[&op] = dag.opToNodeID.lookup(producer);
              }
            }
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
      // modelAndSolveILP(dag);
      MappingSolution solution = modelAndSolveILP(dag);
      const std::vector<int> &chosen = solution.chosenBank;
      DenseMap<Operation *, int> opToBank;
      DenseSet<int> banksInUse;
      for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
        if (chosen[i] < 0) {
          continue;
        }
        Operation *op = dag.nodes[i].op;
        opToBank[op] = chosen[i];
        op->setAttr("bits.bank_id",
                    IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                     chosen[i]));
        op->setAttr(
            "bits.node_id",
            IntegerAttr::get(IntegerType::get(func.getContext(), 64), i));
        op->setAttr(
            "bits.schedule_start",
            IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                             solution.startTime[i]));
        op->setAttr("bits.schedule_end",
                    IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                     solution.endTime[i]));
        op->setAttr(
            "bits.loop_depth",
            IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                             dag.nodes[i].loopDepth));
        if (dag.nodes[i].parentForNode >= 0) {
          op->setAttr("bits.parent_for_node_id",
                      IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                       dag.nodes[i].parentForNode));
        }
        banksInUse.insert(chosen[i]);
      }

      Builder attrBuilder(func.getContext());
      func.walk([&](scf::ForOp forOp) {
        Operation &op = *forOp.getOperation();
        if (loopTripCount.contains(&op)) {
          op.setAttr("bits.trip_count",
                     IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                      loopTripCount.lookup(&op)));
          op.setAttr("bits.best_parallel_banks",
                     IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                      loopBestBanks.lookup(&op)));
          op.setAttr("bits.has_slice_reduction",
                     BoolAttr::get(func.getContext(),
                                   loopHasSliceReduction.lookup(&op)));
          int64_t reduceNode = loopReduceProducerNode.lookup(&op);
          if (reduceNode >= 0) {
            op.setAttr("bits.reduce_op_node_id",
                       IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                        reduceNode));
          }

          auto forNodeIt = dag.opToNodeID.find(&op);
          int forNodeId =
              (forNodeIt != dag.opToNodeID.end()) ? forNodeIt->second : -1;
          if (forNodeId >= 0) {
            DenseSet<int64_t> descendantNodes;
            DenseSet<int64_t> descendantBanks;
            for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
              int parent = dag.nodes[i].parentForNode;
              while (parent >= 0) {
                if (parent == forNodeId) {
                  descendantNodes.insert(i);
                  if (chosen[i] >= 0)
                    descendantBanks.insert(chosen[i]);
                  break;
                }
                parent = dag.nodes[parent].parentForNode;
              }
            }
            SmallVector<int64_t> nodes(descendantNodes.begin(),
                                       descendantNodes.end());
            llvm::sort(nodes);
            SmallVector<Attribute> nodeAttrs;
            for (int64_t node : nodes)
              nodeAttrs.push_back(attrBuilder.getI64IntegerAttr(node));
            op.setAttr("bits.descendant_nodes",
                       ArrayAttr::get(func.getContext(), nodeAttrs));

            SmallVector<int64_t> banks(descendantBanks.begin(),
                                       descendantBanks.end());
            llvm::sort(banks);
            SmallVector<Attribute> bankAttrs;
            for (int64_t b : banks)
              bankAttrs.push_back(attrBuilder.getI64IntegerAttr(b));
            op.setAttr("bits.descendant_banks",
                       ArrayAttr::get(func.getContext(), bankAttrs));
          }

          int64_t baseBank =
              (forNodeId >= 0 &&
               forNodeId < static_cast<int64_t>(chosen.size()))
                  ? chosen[forNodeId]
                  : -1;
          if (baseBank >= 0) {
            int64_t groupSize = std::max<int64_t>(1, loopBestBanks.lookup(&op));
            if (!loopHasSliceReduction.lookup(&op)) {
              groupSize = kNumBanks;
            }
            groupSize = std::min<int64_t>(groupSize, kNumBanks);
            SmallVector<int64_t> bankGroup = buildBankGroup(baseBank, groupSize);
            SmallVector<Attribute> groupAttrs;
            for (int64_t b : bankGroup)
              groupAttrs.push_back(attrBuilder.getI64IntegerAttr(b));
            op.setAttr("bits.bank_group",
                       ArrayAttr::get(func.getContext(), groupAttrs));

            for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
              int parent = dag.nodes[i].parentForNode;
              while (parent >= 0) {
                if (parent == forNodeId) {
                  if (!dag.nodes[i].op->hasAttr("bits.bank_group")) {
                    dag.nodes[i].op->setAttr("bits.bank_group",
                                             ArrayAttr::get(func.getContext(),
                                                            groupAttrs));
                  } else {
                    dag.nodes[i].op->setAttr("bits.owner_bank_group",
                                             ArrayAttr::get(func.getContext(),
                                                            groupAttrs));
                  }
                  break;
                }
                parent = dag.nodes[parent].parentForNode;
              }
            }

            int64_t reduceNode = loopReduceProducerNode.lookup(&op);
            if (reduceNode >= 0 &&
                reduceNode < static_cast<int64_t>(dag.nodes.size())) {
              DenseMap<int64_t, int64_t> counts =
                  buildTreeReduceCountPerBank(baseBank, groupSize);
              SmallVector<Attribute> countAttrs;
              SmallVector<int64_t> banks;
              for (auto &it : counts)
                banks.push_back(it.first);
              llvm::sort(banks);
              for (int64_t bankId : banks) {
                NamedAttrList entry;
                entry.set("bank_id", attrBuilder.getI64IntegerAttr(bankId));
                entry.set("reduce_count",
                          attrBuilder.getI64IntegerAttr(counts.lookup(bankId)));
                countAttrs.push_back(
                    DictionaryAttr::get(func.getContext(), entry));
              }
              dag.nodes[reduceNode].op->setAttr(
                  "bits.tree_reduce_bank_counts",
                  ArrayAttr::get(func.getContext(), countAttrs));
              dag.nodes[reduceNode].op->setAttr(
                  "bits.reduce_output_bank",
                  attrBuilder.getI64IntegerAttr(
                      bankGroup.empty() ? baseBank : bankGroup.front()));
            }
          }
        }
      });

      if (kEnableVerboseMappingLog) {
        std::cout << "Bank assignments (operator -> banks):\n";
        for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
          Operation *op = dag.nodes[i].op;
          std::cout << "Op " << i << " (" << op->getName().getStringRef().str()
                    << ") -> ";
          if (auto group =
                  dyn_cast_or_null<ArrayAttr>(op->getAttr("bits.bank_group"))) {
            std::cout << "[";
            bool first = true;
            for (Attribute attr : group) {
              if (!first)
                std::cout << ", ";
              first = false;
              std::cout << cast<IntegerAttr>(attr).getInt();
            }
            std::cout << "]";
          } else if (chosen[i] >= 0) {
            std::cout << "[" << chosen[i] << "]";
          } else {
            std::cout << "[]";
          }
          std::cout << "\n";
        }
        std::cout << "For-region planning summary:\n";
      }
      SmallVector<scf::ForOp> allForOps;
      func.walk([&](scf::ForOp forOp) { allForOps.push_back(forOp); });
      llvm::sort(allForOps, [](scf::ForOp a, scf::ForOp b) {
        auto depthA = dyn_cast_or_null<IntegerAttr>(a->getAttr("bits.loop_depth"));
        auto depthB = dyn_cast_or_null<IntegerAttr>(b->getAttr("bits.loop_depth"));
        int64_t da = depthA ? depthA.getInt() : 0;
        int64_t db = depthB ? depthB.getInt() : 0;
        return da < db;
      });

      for (scf::ForOp forOp : allForOps) {
        auto tripMaybe = getConstTripCount(forOp);
        if (failed(tripMaybe))
          continue;
        int64_t tripCount = *tripMaybe;
        int64_t baseBank = -1;
        if (auto b = dyn_cast_or_null<IntegerAttr>(
                forOp->getAttr("bits.bank_id"))) {
          baseBank = b.getInt();
        }
        auto groupAttr =
            dyn_cast_or_null<ArrayAttr>(forOp->getAttr("bits.bank_group"));
        int64_t availBanks = groupAttr ? groupAttr.size() : 1;
        if (auto parentNodeAttr =
                dyn_cast_or_null<IntegerAttr>(forOp->getAttr("bits.parent_for_node_id"))) {
          int64_t parentNodeId = parentNodeAttr.getInt();
          if (parentNodeId >= 0 &&
              parentNodeId < static_cast<int64_t>(dag.nodes.size())) {
            Operation *parentForOp = dag.nodes[parentNodeId].op;
            if (auto parentRegionCountAttr = dyn_cast_or_null<IntegerAttr>(
                    parentForOp->getAttr("bits.region_count"))) {
              int64_t parentRegions = std::max<int64_t>(1, parentRegionCountAttr.getInt());
              availBanks = std::max<int64_t>(1, availBanks / parentRegions);
            }
          }
        }
        if (baseBank < 0)
          baseBank = 0;

        // If this for directly contains one nested for, optimize region split for
        // outer and leaf-plan the inner.
        scf::ForOp childFor;
        int childForCount = 0;
        for (Operation &op : forOp.getBody()->without_terminator()) {
          if (auto nested = dyn_cast<scf::ForOp>(&op)) {
            childFor = nested;
            ++childForCount;
          }
        }

        RegionPlan chosenPlan;
        if (childForCount == 1) {
          int64_t bestCost = std::numeric_limits<int64_t>::max();
          auto candidates =
              getPowerOfTwoChoices(std::min<int64_t>(tripCount, availBanks));
          for (int64_t regions : candidates) {
            int64_t banksPerRegion = std::max<int64_t>(1, availBanks / regions);
            RegionPlan innerPlan =
                planLeafForRegions(childFor, baseBank, banksPerRegion);
            int64_t iterPerRegion = std::max<int64_t>(1, tripCount / regions);
            int64_t cost = iterPerRegion * innerPlan.estimatedMakespan;
            if (cost < bestCost) {
              bestCost = cost;
              chosenPlan.regionCount = regions;
              chosenPlan.banksPerRegion = banksPerRegion;
              chosenPlan.iterPerRegion = iterPerRegion;
              chosenPlan.unrollFactor = innerPlan.unrollFactor;
              chosenPlan.computeOpsPerRegion = innerPlan.computeOpsPerRegion;
              chosenPlan.estimatedMakespan = bestCost;
              chosenPlan.regionBanks.clear();
              for (int64_t r = 0; r < regions; ++r) {
                SmallVector<int64_t> banks;
                for (int64_t i = 0; i < banksPerRegion; ++i)
                  banks.push_back((baseBank + r * banksPerRegion + i) %
                                  kNumBanks);
                chosenPlan.regionBanks.push_back(std::move(banks));
              }
            }
          }
          forOp->setAttr(
              "bits.inner_for_unroll_factor",
              attrBuilder.getI64IntegerAttr(chosenPlan.unrollFactor));
          forOp->setAttr(
              "bits.inner_for_compute_ops_per_region",
              attrBuilder.getI64IntegerAttr(chosenPlan.computeOpsPerRegion));
        } else {
          chosenPlan = planLeafForRegions(forOp, baseBank, availBanks);
        }

        SmallVector<Attribute> regionAttrs;
        for (const auto &banks : chosenPlan.regionBanks) {
          SmallVector<Attribute> bankAttrs;
          for (int64_t b : banks)
            bankAttrs.push_back(attrBuilder.getI64IntegerAttr(b));
          regionAttrs.push_back(ArrayAttr::get(func.getContext(), bankAttrs));
        }
        forOp->setAttr("bits.region_count",
                       attrBuilder.getI64IntegerAttr(chosenPlan.regionCount));
        forOp->setAttr("bits.region_available_banks",
                       attrBuilder.getI64IntegerAttr(availBanks));
        forOp->setAttr("bits.region_banks",
                       ArrayAttr::get(func.getContext(), regionAttrs));

        if (kEnableVerboseMappingLog) {
          int64_t nodeId = -1;
          if (auto nid = dyn_cast_or_null<IntegerAttr>(forOp->getAttr("bits.node_id")))
            nodeId = nid.getInt();
          std::cout << "  for(node_id=" << nodeId << "): regions="
                    << chosenPlan.regionCount
                    << ", avail_banks=" << availBanks
                    << ", banks_per_region=" << chosenPlan.banksPerRegion
                    << ", inner_unroll=" << chosenPlan.unrollFactor
                    << ", compute_ops_per_region="
                    << chosenPlan.computeOpsPerRegion << "\n";
        }
      }

      DenseMap<Value, int64_t> valueIds;
      int64_t nextValueId = 0;
      func.walk([&](Operation *op) {
        if (isa<func::FuncOp>(op))
          return;
        if (op->getNumResults() == 0) {
          return;
        }
        SmallVector<Attribute> ids;
        ids.reserve(op->getNumResults());
        for (Value result : op->getResults()) {
          valueIds[result] = nextValueId;
          ids.push_back(IntegerAttr::get(
              IntegerType::get(func.getContext(), 64), nextValueId));
          ++nextValueId;
        }
        op->setAttr("bits.value_ids", ArrayAttr::get(func.getContext(), ids));
      });

      SmallVector<Attribute> reduceOutputs;
      for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
        Operation *op = dag.nodes[i].op;
        auto countsAttr =
            dyn_cast_or_null<ArrayAttr>(op->getAttr("bits.tree_reduce_bank_counts"));
        auto outBankAttr =
            dyn_cast_or_null<IntegerAttr>(op->getAttr("bits.reduce_output_bank"));
        if (!countsAttr || !outBankAttr)
          continue;

        int64_t valueId = -1;
        if (op->getNumResults() > 0)
          valueId = valueIds.lookup(op->getResult(0));
        NamedAttrList out;
        out.set("node_id", attrBuilder.getI64IntegerAttr(i));
        out.set("op_name",
                StringAttr::get(func.getContext(), op->getName().getStringRef()));
        out.set("output_bank",
                attrBuilder.getI64IntegerAttr(outBankAttr.getInt()));
        if (valueId >= 0)
          out.set("value_id", attrBuilder.getI64IntegerAttr(valueId));
        out.set("bank_reduce_counts", countsAttr);
        reduceOutputs.push_back(DictionaryAttr::get(func.getContext(), out));
      }
      func->setAttr("bits.reduce_outputs",
                    ArrayAttr::get(func.getContext(), reduceOutputs));

      DenseMap<int, DenseSet<int64_t>> subgraphInputs;
      DenseMap<int, DenseSet<int64_t>> subgraphOutputs;
      DenseMap<int, SmallVector<int64_t>> subgraphNodes;
      SmallVector<TransferEvent> transferEvents;

      for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
        int bank = chosen[i];
        if (bank < 0) {
          continue;
        }
        subgraphNodes[bank].push_back(i);
        Operation *mappedOp = dag.nodes[i].op;
        for (Value operand : mappedOp->getOperands()) {
          bool internal = false;
          if (auto *defOp = operand.getDefiningOp()) {
            auto it = opToBank.find(defOp);
            if (it != opToBank.end() && it->second == bank) {
              internal = true;
            }
            auto predNodeIt = dag.opToNodeID.find(defOp);
            if (it != opToBank.end() && predNodeIt != dag.opToNodeID.end() &&
                it->second != bank) {
              int predNode = predNodeIt->second;
              int64_t bitwidth = getSliceBitwidth(operand);
              int64_t transferCost = kCloneCostPerBit * bitwidth;
              transferEvents.push_back(TransferEvent{
                  predNode, i, it->second, bank, solution.endTime[predNode],
                  solution.endTime[predNode] + transferCost,
                  valueIds.lookup(operand)});
            }
          }
          if (!internal) {
            subgraphInputs[bank].insert(valueIds.lookup(operand));
          }
        }

        for (Value result : mappedOp->getResults()) {
          bool isOutput = false;
          for (OpOperand &use : result.getUses()) {
            Operation *user = use.getOwner();
            auto it = opToBank.find(user);
            if (it != opToBank.end() && it->second == bank) {
              continue;
            }
            isOutput = true;
            break;
          }
          if (isOutput) {
            subgraphOutputs[bank].insert(valueIds.lookup(result));
          }
        }
      }

      Builder builder(func.getContext());
      SmallVector<Attribute> subgraphAttrs;
      for (int bank : banksInUse) {
        SmallVector<Attribute> nodeAttrs;
        for (int64_t nodeId : subgraphNodes[bank]) {
          nodeAttrs.push_back(builder.getI64IntegerAttr(nodeId));
        }
        SmallVector<int64_t> inputIds(subgraphInputs[bank].begin(),
                                      subgraphInputs[bank].end());
        llvm::sort(inputIds);
        SmallVector<Attribute> inputAttrs;
        for (int64_t valueId : inputIds) {
          inputAttrs.push_back(builder.getI64IntegerAttr(valueId));
        }
        SmallVector<int64_t> outputIds(subgraphOutputs[bank].begin(),
                                       subgraphOutputs[bank].end());
        llvm::sort(outputIds);
        SmallVector<Attribute> outputAttrs;
        for (int64_t valueId : outputIds) {
          outputAttrs.push_back(builder.getI64IntegerAttr(valueId));
        }
        NamedAttrList subgraph;
        subgraph.set("bank_id", builder.getI64IntegerAttr(bank));
        subgraph.set("nodes", ArrayAttr::get(func.getContext(), nodeAttrs));
        subgraph.set("inputs", ArrayAttr::get(func.getContext(), inputAttrs));
        subgraph.set("outputs", ArrayAttr::get(func.getContext(), outputAttrs));
        subgraphAttrs.push_back(
            DictionaryAttr::get(func.getContext(), subgraph));
      }
      func->setAttr("bits.subgraphs",
                    ArrayAttr::get(func.getContext(), subgraphAttrs));
      SmallVector<Attribute> bankPlans;
      bankPlans.reserve(kNumBanks);
      int64_t totalAdjustedEvents = 0;
      DenseMap<int64_t, DenseSet<int64_t>> managedNodeSets;
      for (int i = 0; i < static_cast<int>(dag.nodes.size()); ++i) {
        int current = i;
        while (current >= 0) {
          int ownerBank = chosen[current];
          if (ownerBank >= 0)
            managedNodeSets[ownerBank].insert(i);
          current = dag.nodes[current].parentForNode;
        }
      }
      for (int64_t bank = 0; bank < kNumBanks; ++bank) {
        SmallVector<Attribute> nodeAttrs;
        struct SegmentInfo {
          int64_t start = 0;
          int64_t end = 0;
          StringRef kind;
          int64_t nodeId = -1;
          int64_t valueId = -1;
          int64_t peerBank = -1;
        };
        SmallVector<SegmentInfo> segments;
        if (subgraphNodes.contains(bank)) {
          for (int64_t nodeId : subgraphNodes[bank]) {
            nodeAttrs.push_back(builder.getI64IntegerAttr(nodeId));
            segments.push_back(SegmentInfo{solution.startTime[nodeId],
                                           solution.endTime[nodeId], "compute",
                                           nodeId, -1, -1});
          }
        }
        for (const TransferEvent &evt : transferEvents) {
          if (evt.srcBank == bank) {
            segments.push_back(SegmentInfo{evt.start, evt.end, "send", evt.srcNodeId,
                                           evt.valueId, evt.dstBank});
          }
          if (evt.dstBank == bank) {
            segments.push_back(SegmentInfo{evt.start, evt.end, "recv", evt.dstNodeId,
                                           evt.valueId, evt.srcBank});
          }
        }
        llvm::sort(segments, [&](const SegmentInfo &lhs, const SegmentInfo &rhs) {
          if (lhs.start != rhs.start)
            return lhs.start < rhs.start;
          return lhs.end < rhs.end;
        });

        SmallVector<Attribute> segmentAttrs;
        int64_t cursor = 0;
        for (const SegmentInfo &seg : segments) {
          int64_t originalStart = seg.start;
          int64_t originalEnd = seg.end;
          int64_t duration = std::max<int64_t>(0, originalEnd - originalStart);
          int64_t start = std::max(originalStart, cursor);
          int64_t end = start + duration;
          if (start != originalStart)
            ++totalAdjustedEvents;
          if (start > cursor) {
            NamedAttrList idleSeg;
            idleSeg.set("kind", StringAttr::get(func.getContext(), "idle"));
            idleSeg.set("start", builder.getI64IntegerAttr(cursor));
            idleSeg.set("end", builder.getI64IntegerAttr(start));
            segmentAttrs.push_back(DictionaryAttr::get(func.getContext(), idleSeg));
          }
          NamedAttrList eventSeg;
          eventSeg.set("kind", StringAttr::get(func.getContext(), seg.kind));
          eventSeg.set("start", builder.getI64IntegerAttr(start));
          eventSeg.set("end", builder.getI64IntegerAttr(end));
          eventSeg.set("node_id", builder.getI64IntegerAttr(seg.nodeId));
          if (seg.valueId >= 0)
            eventSeg.set("value_id", builder.getI64IntegerAttr(seg.valueId));
          if (seg.peerBank >= 0)
            eventSeg.set("peer_bank", builder.getI64IntegerAttr(seg.peerBank));
          eventSeg.set("original_start",
                       builder.getI64IntegerAttr(originalStart));
          eventSeg.set("original_end", builder.getI64IntegerAttr(originalEnd));
          segmentAttrs.push_back(
              DictionaryAttr::get(func.getContext(), eventSeg));
          cursor = std::max(cursor, end);
        }

        NamedAttrList bankPlan;
        bankPlan.set("bank_id", builder.getI64IntegerAttr(bank));
        bankPlan.set("nodes", ArrayAttr::get(func.getContext(), nodeAttrs));
        SmallVector<int64_t> managedNodes;
        if (managedNodeSets.contains(bank)) {
          managedNodes.assign(managedNodeSets[bank].begin(),
                              managedNodeSets[bank].end());
          llvm::sort(managedNodes);
        }
        SmallVector<Attribute> managedNodeAttrs;
        for (int64_t nodeId : managedNodes)
          managedNodeAttrs.push_back(builder.getI64IntegerAttr(nodeId));
        bankPlan.set("managed_nodes",
                     ArrayAttr::get(func.getContext(), managedNodeAttrs));
        bankPlan.set("segments",
                     ArrayAttr::get(func.getContext(), segmentAttrs));
        bankPlans.push_back(DictionaryAttr::get(func.getContext(), bankPlan));
      }
      func->setAttr("bits.bank_plan",
                    ArrayAttr::get(func.getContext(), bankPlans));
      func->setAttr("bits.bank_plan_adjusted_events",
                    IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                     totalAdjustedEvents));
      func->setAttr("bits.bank_plan_ready",
                    BoolAttr::get(func.getContext(), true));
      func->setAttr("bits.mapping_progress_step",
                    IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                     5));
    } catch (GRBException &e) {
      std::cerr << "Gurobi error: " << e.getMessage() << "\n";
    } catch (std::exception &ex) {
      std::cerr << "Error: " << ex.what() << "\n";
    }
  }
};

} // namespace mlir::bits
