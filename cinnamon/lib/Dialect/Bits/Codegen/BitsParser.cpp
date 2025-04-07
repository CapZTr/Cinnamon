#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <vector>

using namespace mlir;
using namespace mlir::bits;

BitsParser::BitsParser(ModuleOp module) : module(module) {}

void BitsParser::parse() {
  SmallVector<AddOp> pendingAdds;

  module->walk([&](Operation *op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      llvm::errs() << "Parsing transpose\n";
      parseTranspose(transpose);
    } else if (auto add = dyn_cast<AddOp>(op)) {
      llvm::errs() << "Adding add\n";
      pendingAdds.push_back(add);
    }
  });

  bool progress = true;
  while (progress && !pendingAdds.empty()) {
    progress = false;

    for (auto it = pendingAdds.begin(); it != pendingAdds.end();) {
      if (operandsParsed(*it)) {
        llvm::errs() << "Parsing add\n";
        parseAdd(*it);
        it = pendingAdds.erase(it);
        progress = true;
      } else {
        ++it;
      }
    }
  }

  if (!pendingAdds.empty()) {
    llvm::errs() << "BitsParser: Some AddOps could not be parsed due to missing operands.\n";
  }
}

void BitsParser::parseTranspose(TransposeOp op) {
  auto input = op.getInput();
  auto result = op.getOutput();
  auto type = dyn_cast<RankedTensorType>(input.getType());

  int64_t bitWidth = type.getElementTypeBitWidth();
  int64_t vectorLen = type.getNumElements();
  BitplaneData data{bitWidth, vectorLen};

  // parsed[input] = data;
  parsed[result] = data;
  inputs[input] = data;
}

void BitsParser::parseAdd(AddOp op) {
  auto lhsData = parsed.lookup(op.getLhs());
  auto rhsData = parsed.lookup(op.getRhs());

  auto result = op.getResult();
  auto type = dyn_cast<SliceType>(result.getType());
  auto resultData = BitplaneData{type.getBitWidth(), type.getVectorLength()};

  parsed[result] = resultData;
  adds.push_back(AddData{lhsData, rhsData, resultData});
}

bool BitsParser::operandsParsed(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!parsed.contains(operand))
      return false;
  }
  return true;
}
