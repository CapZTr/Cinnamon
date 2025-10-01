#include "cinm-mlir/Dialect/PuD/Codegen/OperandTracker.h"

#include <cassert>
#include <memory>
#include <optional>
#include <variant>


OperandTracker::OperandTracker() {
  bGroupExpressions.resize(16);
}

std::unique_ptr<OperandExpr> OperandTracker::executeAP(int index) {
  assert(index >= 12 && index <= 15);
  int index1, index2, index3;
  switch (index) {
    case 12:
      index1 = 0;
      index2 = 1;
      index3 = 2;
      break;
    case 13:
      index1 = 1;
      index2 = 2;
      index3 = 3;
      break;
    case 14:
      index1 = 4;
      index2 = 1;
      index3 = 2;
      break;
    case 15:
      index1 = 6;
      index2 = 0;
      index3 = 3;
      break;
    default:
      assert(false && "This should never happen");
      break;
  }
  auto m = doMaj(index1, index2, index3);
  bGroupExpressions[index] = m->clone();
  return m;
}

std::unique_ptr<OperandExpr> OperandTracker::executeAAP(
    std::variant<OperandExpr *, int> source,
    std::optional<int> destination) {
  OperandExpr *toClone;
  std::unique_ptr<OperandExpr> temp;
  if (auto addr1 = std::get_if<int>(&source)) {
    if (*addr1 >= 12 && *addr1 <= 15) {
      temp = executeAP(*addr1);
      toClone = temp.get();
    } else {
      assert(*addr1 <= 7);
      toClone = bGroupExpressions[*addr1].get();
    }
  } else {
    toClone = std::get<OperandExpr *>(source);
  }

  if (!destination.has_value()) {
    return toClone->clone();
  } else {
    auto addr2 = *destination;
    bGroupExpressions[addr2] = toClone->clone();
    if (addr2 <= 7) {
      if (addr2 >= 4) {
        doNot(addr2);
      }
    } else {
      switch (addr2) {
        case 8:
          bGroupExpressions[0] = toClone->clone();
          bGroupExpressions[5] = toClone->clone();
          doNot(5);
          break;
        case 9:
          bGroupExpressions[1] = toClone->clone();
          bGroupExpressions[7] = toClone->clone();
          doNot(7);
          break;
        case 10:
          bGroupExpressions[2] = toClone->clone();
          bGroupExpressions[3] = toClone->clone();
          break;
        case 11:
          bGroupExpressions[0] = toClone->clone();
          bGroupExpressions[3] = toClone->clone();
          break;
        case 12:
          bGroupExpressions[0] = toClone->clone();
          bGroupExpressions[1] = toClone->clone();
          bGroupExpressions[2] = toClone->clone();
          break;
        case 13:
          bGroupExpressions[1] = toClone->clone();
          bGroupExpressions[2] = toClone->clone();
          bGroupExpressions[3] = toClone->clone();
          break;
        case 14:
          bGroupExpressions[4] = toClone->clone();
          bGroupExpressions[1] = toClone->clone();
          bGroupExpressions[2] = toClone->clone();
          doNot(4);
          break;
        case 15:
          bGroupExpressions[6] = toClone->clone();
          bGroupExpressions[0] = toClone->clone();
          bGroupExpressions[3] = toClone->clone();
          doNot(6);
          break;
        default:
          break;
      }
    }
    return nullptr;
  }
}

std::unique_ptr<OperandExpr> OperandTracker::doMaj(
    int index1, int index2, int index3) {
  assert(bGroupExpressions[index1] && bGroupExpressions[index2]
      && bGroupExpressions[index3]
      && "doMaj operands must be initialized before use");
  auto m = std::make_unique<MajExpr>(
      bGroupExpressions[index1]->clone(),
      bGroupExpressions[index2]->clone(),
      bGroupExpressions[index3]->clone());
  bGroupExpressions[index1] = m->clone();
  bGroupExpressions[index2] = m->clone();
  bGroupExpressions[index3] = m->clone();
  if (index1 == 4 || index1 == 6) {
    doNot(index1);
  }
  return m;
}

void OperandTracker::doNot(int index) {
  assert(index >= 4 && index <= 7);
  auto n = std::make_unique<NotExpr>(bGroupExpressions[index]->clone());
  if (index % 2 == 0) {
    bGroupExpressions[index + 1] = std::move(n);
  } else {
    bGroupExpressions[index - 1] = std::move(n);
  }
}
