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
int cloneDelay = 1;

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
    model.addConstr(end[i] == start[i] + 1, "duration_" + std::to_string(i));
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

  GRBLinExpr totalClones = 0;
  const int M = 1000000;
  for (int i = 0; i < N; ++i) {
    for (int j : dag.nodes[i].succs) {
      GRBVar cloneVar = model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
          "clone_" + std::to_string(i) + "_" + std::to_string(j));
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
      // model.addConstr(start[j] >= end[i] + 1,
          "sched_dep_" + std::to_string(i) + "_" + std::to_string(j));
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
    for (int n = 0; n < numBanks; ++i) {
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
