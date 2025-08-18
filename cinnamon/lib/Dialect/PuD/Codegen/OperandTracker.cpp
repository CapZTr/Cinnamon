#include "cinm-mlir/Dialect/PuD/Codegen/OperandTracker.h"

#include <cassert>
#include <memory>
#include <optional>
#include <variant>


OperandTracker::OperandTracker() {
  for (int i = 0; i < 16; ++i) {
    bGroupExpressions.push_back(nullptr);
  }
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
  if (auto addr1 = std::get_if<int>(&source)) {
    if (*addr1 >= 12 && *addr1 <= 15) {
      toClone = executeAP(*addr1).get();
    } else {
      assert(*addr1 <= 6 && *addr1 != 5);
      toClone = bGroupExpressions[*addr1].get();
    }
  } else {
    toClone = std::get<OperandExpr *>(source);
  }

  if (!destination.has_value()) {
    return toClone->clone();
  } else {
    auto addr2 = *destination;
    assert(addr2 <= 11);
    if (addr2 <= 7) {
      bGroupExpressions[addr2] = toClone->clone();
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
        default:
          break;
      }
    }
    return nullptr;
  }
}

std::unique_ptr<OperandExpr> OperandTracker::doMaj(int index1, int index2, int index3) {
  auto m = std::make_unique<MajExpr>(
      std::move(bGroupExpressions[index1]),
      std::move(bGroupExpressions[index2]),
      std::move(bGroupExpressions[index3]));
  bGroupExpressions[index1] = m->clone();
  bGroupExpressions[index2] = m->clone();
  bGroupExpressions[index3] = m->clone();
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
