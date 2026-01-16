#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h"

#include <exception>
#include <iostream>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <string>
#include <vector>

#include "gurobi_c++.h"
#include "gurobi_c.h"

namespace mlir::bits {

//===- Generated passes ---------------------------------------------------===//

#define GEN_PASS_DEF_BITSOPTMISEMAPPINGPASS
#include "cinm-mlir/Dialect/Bits/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//

int numBanks = 32;
int cloneDelay = 10;

struct Node {
  AddIOp op;
  llvm::SmallVector<int, 2> preds;
  llvm::SmallVector<int, 4> succs;
};

struct DAG {
  llvm::SmallVector<Node> nodes;
  llvm::DenseMap<Operation *, int> opToNodeID;
};

void modelAndSolveILP(const DAG &dag) {
  const int N = dag.nodes.size();

  GRBEnv env = GRBEnv(true);
  env.start();
  GRBModel model = GRBModel(env);

  std::vector<std::vector<GRBVar>> x(N, std::vector<GRBVar>(numBanks));
  for (int i = 0; i < N; ++i) {
    for (int n = 0; n < numBanks; ++n) {
      x[i][n] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
          "x_" + std::to_string(i) + "_" + std::to_string(n));
    }
  }

  for (int i = 0; i < N; ++i) {
    GRBLinExpr sumBanks = 0;
    for (int n = 0; n < numBanks; ++n) {
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
    model.addConstr(end[i] == start[i] + 9, "duration_" + std::to_string(i));
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
          if (dep[k][j]) dep[i][j] = true;
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

  GRBLinExpr totalClones = 0;
  const int M = 100;
  for (int i = 0; i < N; ++i) {
    for (int j : dag.nodes[i].succs) {
      GRBVar cloneVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
          "clone_" + std::to_string(i) + "_" + std::to_string(j));
      cloneEdges.push_back(CloneEdge{i, j, cloneVar});
      totalClones += cloneVar;
      GRBLinExpr sumB;
      for (int k = 0; k < numBanks; ++k) {
        GRBVar b = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "b_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k));
        model.addConstr(b <= x[i][k]);
        model.addConstr(b <= x[j][k]);
        model.addConstr(b >= x[i][k] + x[j][k] - 1);
        sumB += b;
      }
      model.addConstr(cloneVar + sumB == 1,
          "clone_link_" + std::to_string(i) + "_" + std::to_string(j));
      model.addConstr(start[j] >= end[i] + cloneDelay * cloneVar,
          "sched_dep_" + std::to_string(i) + "_" + std::to_string(j));

      for (int k = 0; k < N; ++k) {
        GRBLinExpr sumBsrc;
        for (int bank = 0; bank < numBanks; ++bank) {
          GRBVar b = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
              "b_src_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                  std::to_string(k) + "_" + std::to_string(bank));
          model.addConstr(b <= x[i][bank]);
          model.addConstr(b <= x[k][bank]);
          model.addConstr(b >= x[i][bank] + x[k][bank] - 1);
          sumBsrc += b;
        }
        GRBVar sameBankSrc = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "same_src_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));
        model.addConstr(sumBsrc == sameBankSrc,
            "same_src_link_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        GRBVar orderSrc = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "order_src_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));
        model.addConstr(end[k] <= end[i] +
                M * (2 - cloneVar - sameBankSrc + orderSrc),
            "clone_src_before_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        model.addConstr(start[k] >= end[i] + cloneDelay -
                M * (2 - cloneVar - sameBankSrc + (1 - orderSrc)),
            "clone_src_after_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));

        GRBLinExpr sumBdst;
        for (int bank = 0; bank < numBanks; ++bank) {
          GRBVar b = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
              "b_dst_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                  std::to_string(k) + "_" + std::to_string(bank));
          model.addConstr(b <= x[j][bank]);
          model.addConstr(b <= x[k][bank]);
          model.addConstr(b >= x[j][bank] + x[k][bank] - 1);
          sumBdst += b;
        }
        GRBVar sameBankDst = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "same_dst_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));
        model.addConstr(sumBdst == sameBankDst,
            "same_dst_link_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        GRBVar orderDst = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "order_dst_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));
        model.addConstr(end[k] <= end[i] +
                M * (2 - cloneVar - sameBankDst + orderDst),
            "clone_dst_before_" + std::to_string(i) + "_" + std::to_string(j) +
                "_" + std::to_string(k));
        model.addConstr(start[k] >= end[i] + cloneDelay -
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
      auto addCloneOrderConstraints = [&](int a, int b,
                                          llvm::StringRef tag) {
        GRBLinExpr sumB;
        for (int bank = 0; bank < numBanks; ++bank) {
          GRBVar bVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
              "b_" + std::string(tag) + "_" + std::to_string(a) + "_" +
                  std::to_string(b) + "_" + std::to_string(bank));
          model.addConstr(bVar <= x[a][bank]);
          model.addConstr(bVar <= x[b][bank]);
          model.addConstr(bVar >= x[a][bank] + x[b][bank] - 1);
          sumB += bVar;
        }
        GRBVar sameBank = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "same_" + std::string(tag) + "_" + std::to_string(a) + "_" +
                std::to_string(b));
        model.addConstr(sumB == sameBank,
            "same_link_" + std::string(tag) + "_" + std::to_string(a) + "_" +
                std::to_string(b));
        GRBVar orderVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "order_" + std::string(tag) + "_" + std::to_string(a) + "_" +
                std::to_string(b));
        model.addConstr(end[c1.src] + cloneDelay <= end[c2.src] +
                M * (3 - c1.active - c2.active - sameBank + orderVar),
            "clone_serial_ab_" + std::string(tag) + "_" + std::to_string(a) +
                "_" + std::to_string(b));
        model.addConstr(end[c2.src] + cloneDelay <= end[c1.src] +
                M * (3 - c1.active - c2.active - sameBank + (1 - orderVar)),
            "clone_serial_ba_" + std::string(tag) + "_" + std::to_string(a) +
                "_" + std::to_string(b));
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
            "order_" + std::to_string(i) + "_" + std::to_string(j));
        GRBVar diffVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
            "diff_" + std::to_string(i) + "_" + std::to_string(j));
        GRBLinExpr sumB;
        for (int k = 0; k < numBanks; ++k) {
          GRBVar b = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
              "b_" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(k));
          model.addConstr(b <= x[i][k]);
          model.addConstr(b <= x[j][k]);
          model.addConstr(b >= x[i][k] + x[j][k] - 1);
          sumB += b;
        }
        model.addConstr(diffVar + sumB == 1,
            "diff_link_" + std::to_string(i) + "_" + std::to_string(j));
        model.addConstr(start[j] >= end[i] - M * (diffVar + 1.0 - orderVar), 
            "serial_ij_" + std::to_string(i) + "_" + std::to_string(j));
        model.addConstr(start[i] >= end[j] - M * (diffVar + orderVar), 
            "serial_ji_" + std::to_string(i) + "_" + std::to_string(j));
      }
    }
  }

  GRBVar makespan = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, "makespan");
  for (int i = 0; i < N; ++i) {
    model.addConstr(makespan >= end[i], "mkspan_ge_end_" + std::to_string(i));
  }

  model.setObjective(makespan + 0, GRB_MINIMIZE);
  model.optimize();

  int status = model.get(GRB_IntAttr_Status);
  if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL) {
    std::cerr << "Solving ended with status: " << status << "\n";
  }

  std::vector<std::pair<int, int>> chosen;
  chosen.reserve(N);

  for (int i = 0; i < N; ++i) {
    for (int n = 0; n < numBanks; ++n) {
      double val = x[i][n].get(GRB_DoubleAttr_X);
      if (val > 0.5) {
        chosen.emplace_back(i, n);
        break;
      }
    }
  }

  std::cout << "Chosen mapping (operator -> bank):\n";
  for (auto [i, n] : chosen) {
    std::cout << "Op " << i << " -> bank " << n << "\n";
  }
}

struct BitsOptimiseMappingPass
    : public impl::BitsOptmiseMappingPassBase<BitsOptimiseMappingPass> {
  using Base::Base;

  void runOnOperation() final {
    func::FuncOp func = getOperation();
    DAG dag;

    Block &block = func.getBody().front();

    for (Operation &op : block) {
      if (auto add = llvm::dyn_cast<AddIOp>(&op)) {
        int id = dag.nodes.size();
        dag.nodes.push_back(Node{add, {}, {}});
        dag.opToNodeID[&op] = id;
      }
    }

    if (dag.nodes.empty()) {
      return;
    }

    for (int id = 0; id < (int)dag.nodes.size(); ++id) {
      AddIOp add = dag.nodes[id].op;
      for (Value operand : add->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          auto it = dag.opToNodeID.find(defOp);
          if (it == dag.opToNodeID.end()) {
            continue;
          }

          int predID = it->second;
          std::cout << predID << " -> " << id << "\n";
          dag.nodes[predID].succs.push_back(id);
          dag.nodes[id].preds.push_back(predID);
        }
      }
    }

    try {
      modelAndSolveILP(dag);
    } catch (GRBException &e) {
      std::cerr << "Gurobi error: " << e.getMessage() << "\n";
    } catch (std::exception &ex) {
      std::cerr << "Error: " << ex.what() << "\n";
    }
  }
};

} // namespace mlir::bits
