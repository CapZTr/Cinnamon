#include "cinm-mlir/Dialect/Bits/Codegen/BitsParser.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>


using namespace mlir;
using namespace mlir::bits;

BitsParser::BitsParser(ModuleOp module) : module(module) {}

void BitsParser::parse() {
  SmallVector<Operation*, 16> pendingBinaryOps;
  llvm::SmallDenseSet<Value, 16> unusedSlices;

  module->walk([&](Operation *op) {
    if (auto transpose = dyn_cast<TransposeOp>(op)) {
      parseTranspose(transpose);
      unusedSlices.insert(transpose.getResult());
    } else if (op->getNumOperands() == 2) {
      pendingBinaryOps.push_back(op);
      unusedSlices.insert(op->getResult(0));
    }
  });

  bool progress = true;
  while (progress && !pendingBinaryOps.empty()) {
    progress = false;

    for (auto it = pendingBinaryOps.begin(); it != pendingBinaryOps.end();) {
      if (operandsParsed(*it)) {
        parseBinaryOp(*it);
        it = pendingBinaryOps.erase(it);
        progress = true;
        for (const auto slice : (*it)->getOperands()) {
          unusedSlices.erase(slice);
        }
      } else {
        ++it;
      }
    }
  }

  if (!pendingBinaryOps.empty()) {
    llvm::errs() << "BitsParser: Some binary ops could not be parsed due to missing operands.\n";
  }

  module->walk([&](func::FuncOp funcOp){
    funcOp->walk([&](func::ReturnOp returnOp) {
      if (returnOp->getNumOperands() == 0) {
        for (const auto slice : unusedSlices) {
          outputs[slice] = parsed.lookup(slice);
        }
      } else {
        for (const auto returnSlice : returnOp->getOperands()) {
          outputs[returnSlice] = parsed.lookup(returnSlice);
        }
      }
    });
  });
}

void BitsParser::parseTranspose(TransposeOp op) {
  auto input = op.getInput();
  auto result = op.getOutput();
  auto type = dyn_cast<RankedTensorType>(input.getType());

  int64_t bitWidth = type.getElementTypeBitWidth();
  int64_t vectorLen = type.getNumElements();
  BitplaneData data{bitWidth, vectorLen, result};

  parsed[result] = data;
  inputs[input] = data;
}

void BitsParser::parseAdd(AddOp op) {
  auto lhsData = parsed.lookup(op.getLhs());
  auto rhsData = parsed.lookup(op.getRhs());

  auto result = op.getResult();
  auto type = dyn_cast<SliceType>(result.getType());
  auto resultData = BitplaneData{type.getBitWidth(), type.getVectorLength(), result};

  parsed[result] = resultData;
  const BinaryOpData addData = BinaryOpData{lhsData, rhsData, resultData};
  binaryOps.push_back(addData);
  adds.push_back(addData);
}

void BitsParser::parseBinaryOp(Operation *op) {
  // auto binaryOp = llvm::dyn_cast<AddOp>(*op);
  // if (!binaryOp) binaryOp = llvm::dyn_cast<SubOp>(*op);

  // if (!binaryOp) {
  //   llvm::errs() << "BitsParser: Unknown type of binary operation.\n";
  //   return;
  // }

  auto lhsData = parsed.lookup(op->getOperand(0));
  auto rhsData = parsed.lookup(op->getOperand(1));

  auto result = op->getResult(0);
  auto type = dyn_cast<SliceType>(result.getType());
  auto resultData = BitplaneData{type.getBitWidth(), type.getVectorLength(), result};

  parsed[result] = resultData;
  const BinaryOpData data = BinaryOpData{lhsData, rhsData, resultData};
  binaryOps.push_back(data);

  if (auto add = dyn_cast<AddOp>(*op)) {
    adds.push_back(data);
  } else if (auto sub = dyn_cast<SubOp>(*op)) {
    subs.push_back(data);
  } else {
    llvm::errs() << "BitsParser: Unknown type of binary operation.\n";
  }
}

void BitsParser::parseSub(SubOp op) {
  auto lhsData = parsed.lookup(op.getLhs());
  auto rhsData = parsed.lookup(op.getRhs());

  auto result = op.getResult();
  auto type = dyn_cast<SliceType>(result.getType());
  auto resultData = BitplaneData{type.getBitWidth(), type.getVectorLength(), result};

  parsed[result] = resultData;
  const BinaryOpData subData = BinaryOpData{lhsData, rhsData, resultData};
  binaryOps.push_back(subData);
  subs.push_back(subData);
}

bool BitsParser::operandsParsed(Operation *op) const {
  for (auto operand : op->getOperands()) {
    if (!parsed.contains(operand))
      return false;
  }
  return true;
}
