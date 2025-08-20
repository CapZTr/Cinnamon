#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "cinm-mlir/Dialect/PuD/Codegen/OperandTracker.h"

#include <algorithm>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SourceMgr.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LLVM.h>

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

const int NUM_RANK_PER_CHANNEL = 2;
const int NUM_BANK_PER_RANK = 8;
const int NUM_SUBARRAY_PER_BANK = 64;
const int NUM_ROW_PER_SUBARRAY = 1024;
const int64_t NUM_ROW_PER_BANK = NUM_ROW_PER_SUBARRAY * NUM_SUBARRAY_PER_BANK;
const int64_t NUM_ROW_PER_RANK = NUM_ROW_PER_BANK * NUM_BANK_PER_RANK;
const int64_t NUM_ROW_PER_CHANEL = NUM_ROW_PER_RANK * NUM_RANK_PER_CHANNEL;
const std::string DUMMY_DATA = "b186649dd2c40617e1df8669b90acd6389c0e5f8e5c059c5a4ea4f9eb6409eaacf4380666a43bcc792e0d3f2a7b88eca6067d625801408a3df929bb8b4136b68";

std::string intToHex(const int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

// For simple test
std::array<uint32_t, 4> inputs = {12345, 45678, 56789, 67890};
std::array<std::bitset<32>, 4> inputBits;
void initTest() {
  for (int i = 0; i < 4; ++i) {
    std::bitset<32> bs(inputs[i]);
    inputBits[i] = bs;
  }
}
llvm::StringMap<std::unique_ptr<OperandExpr>> exprMap;
void storeAndInitInput(std::string baseAddr, int inputIdx) {
  unsigned long long base = std::stoull(baseAddr, nullptr, 0);
  const auto bs = inputBits[inputIdx];
  for (int i = 0; i < 32; ++i) {
    auto addr = intToHex(base + i);
    auto name = std::format("I{}_{}", inputIdx, i);
    auto expr = std::make_unique<DataExpr>(name, bs[31 - i]);
    assert(!exprMap.contains(addr));
    exprMap[addr] = std::move(expr);
  }
}
void loadAndEvaluateResult(std::string baseAddr) {
  unsigned long long base = std::stoull(baseAddr, nullptr, 0);
  std::bitset<32> bs;
  for (int i = 0; i < 32; ++i) {
    auto addr = intToHex(base + i);
    assert(exprMap.contains(addr));
    auto expr = exprMap[addr].get();
    if (expr->evaluate()) {
      bs.set(31 - i);
    }
  }
  std::cout << "DRAM result: " << bs.to_ullong() << "\n\n";
}
const auto cExpr0 = std::make_unique<ConstantExpr>(false);
const auto cExpr1 = std::make_unique<ConstantExpr>(true);

std::string getTraceLine(const int cycle, const std::string &opName,
    const std::string &addr0, const std::optional<std::string> addr1) {
  std::string line = std::format("{} {} {} {} 0", cycle, opName, addr0, DUMMY_DATA);
  return addr1.has_value() ? std::format("{} {}", line, *addr1) : line;
}

int main(int argc, char **argv) {
  MLIRContext context;
  DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);

  registry.insert<mlir::bits::BitsDialect, pud::PuDDialect>();
  
  llvm::SourceMgr srcMgr;
  auto buffer = openInputFile(argv[1]);
  if (!buffer) {
    std::cerr << "Failed to open input file\n";
    return 1;
  }

  srcMgr.AddNewSourceBuffer(std::move(buffer), SMLoc());
  auto module = parseSourceFile<ModuleOp>(srcMgr, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR module\n";
    return 1;
  }

  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(bits_frontend::createConvertArithToBitsPass());
  pm.addNestedPass<func::FuncOp>(mlir::bits::createConvertBitsToPuDPass(true));

  if(failed(pm.run(*module))) {
    std::cerr << "Failed to run passes\n";
    return 1;
  }

  std::vector<std::string> trace;
  trace.push_back(">");

  DenseMap<mlir::TypedValue<IntegerType>, int64_t> indices;
  DenseMap<mlir::TypedValue<pud::RowType>, std::string> addrs;
  DenseMap<mlir::TypedValue<pud::RowType>, int64_t> saMap;
  
  int cycle = 1;

  auto getAddressAsStr = [&indices, &addrs, &saMap](mlir::TypedValue<pud::RowType> rowAddr)
      -> std::string {
    if (!addrs.contains(rowAddr)) {
      assert(!saMap.contains(rowAddr));
      auto op = rowAddr.getDefiningOp();
      assert(isa<pud::GetRowOp>(*op));
      auto getRow = cast<pud::GetRowOp>(*op);
      auto channel = indices.lookup(getRow.getChannelID());
      auto rank = indices.lookup(getRow.getRankID());
      auto bank = indices.lookup(getRow.getBankID());
      auto subarray = indices.lookup(getRow.getSubarrayID());
      auto row = indices.lookup(getRow.getRowID());
      auto group = rowAddr.getType().getGroup();
      if (group == 0) {
        row += 1008;
      } else if (group == 1) {
        row += 1006;
      } else {
        assert(group == 2);
      }
      auto firstRowInSa = channel * NUM_ROW_PER_CHANEL
          + rank * NUM_ROW_PER_RANK
          + bank * NUM_ROW_PER_BANK
          + subarray * NUM_ROW_PER_SUBARRAY;
      saMap[rowAddr] = firstRowInSa;
      addrs[rowAddr] = intToHex(firstRowInSa + row);
    }
    return addrs.lookup(rowAddr);
  };

  auto getRowIndex = [&](mlir::TypedValue<pud::RowType> rowAddr) -> int {
    auto op = rowAddr.getDefiningOp();
    assert(isa<pud::GetRowOp>(*op));
    auto getRow = cast<pud::GetRowOp>(*op);
    return indices.lookup(getRow.getRowID());
  };

  auto getRowNum = [&](mlir::TypedValue<pud::RowType> rowAddr) -> int {
    auto row = getRowIndex(rowAddr);
    assert(row >= 0 && row <= 15);
    return row < 8 ? 1 
        : row < 12 ? 2
        : 3;
  };

  uint64_t sum = 0;
  for (const auto i : inputs) {
    sum += i;
  }
  std::cout << "CPU  result: " << sum << "\n\n";

  initTest();
  OperandTracker tracker;

  module->walk([&](func::FuncOp func) {
    int inputIdx = 0;
    func->walk([&](Operation *op) {
      if (auto cnst = dyn_cast<arith::ConstantOp>(*op)) {
        assert(cnst.getType().isInteger(64));
        indices[cast<mlir::TypedValue<IntegerType>>(cnst.getResult())] = cast<IntegerAttr>
            (cnst.getValue()).getValue().getZExtValue();
      } else if (auto store = dyn_cast<pud::StoreOp>(*op)) {
        auto firstRow = store.getFirstRow();
        auto addr = getAddressAsStr(firstRow);
        storeAndInitInput(addr, inputIdx);
        inputIdx++;
      } else if (auto load = dyn_cast<pud::LoadOp>(*op)) {
        auto firstRow = load.getFirstRow();
        auto addr = getAddressAsStr(firstRow);
        loadAndEvaluateResult(addr);
      } else if (auto ap = dyn_cast<pud::APOp>(*op)) {
        auto row = ap.getAddr();
        assert(row.getType().getGroup() == 0);
        auto addr = getAddressAsStr(row);
        auto rowNum = getRowNum(row);
        const std::string opName = rowNum == 1 ? "S"
            : rowNum == 2 ? "D"
            : "T";
        trace.push_back(getTraceLine(cycle, opName, addr, std::nullopt));
        cycle++;
        // Tracking
        tracker.executeAP(getRowIndex(row));
        // std::cout << "===== AP =====\n";
        // std::cout << "AP " << addr << ": " << tracker.getBGroupExprs()[getRowIndex(row)]->evaluate() << "\n";
        // std::cout << "==============\n";
      } else if (auto aap = dyn_cast<pud::AAPOp>(*op)) {
        // std::cout << "=====AAP =====\n";
        auto row0 = aap.getSrcAddr();
        auto addr0 = getAddressAsStr(row0);
        int rowNum0 = 1;
        if (row0.getType().getGroup() == 0)
          rowNum0 = getRowNum(row0);
        auto row1 = aap.getDstAddr();
        auto addr1 = getAddressAsStr(row1);
        int rowNum1 = 1;
        if (row1.getType().getGroup() == 0)
          rowNum1 = getRowNum(row1);
        if (saMap.lookup(row0) == saMap.lookup(row1)) {
          auto maxNum = std::max(rowNum0, rowNum1);
          const std::string opName = maxNum == 1 ? "O"
              : maxNum == 2 ? "ODRA"
              : "OTRA";
          trace.push_back(getTraceLine(cycle, opName, addr0, std::make_optional(addr1)));
          cycle++;
        } else {
          trace.push_back(getTraceLine(cycle, "R", addr0, std::nullopt));
          cycle++;
          trace.push_back(getTraceLine(cycle, "W", addr1, std::nullopt));
          cycle++;
        }
        // Tracking
        if (row0.getType().getGroup() == 0) {
          auto index0 = getRowIndex(row0);
          if (row1.getType().getGroup() == 0) {
            auto index1 = getRowIndex(row1);
            tracker.executeAAP(index0, index1);
            // std::cout << "Source: " << addr0 << " = " << tracker.getBGroupExprs()[index0]->evaluate() << "\n";
            // std::cout << "Destination: " << addr1 << " = " << tracker.getBGroupExprs()[index1]->evaluate() << "\n";
          } else {
            assert(row1.getType().getGroup() == 2);
            auto expr = tracker.executeAAP(index0, std::nullopt);
            // std::cout << "Source: " << addr0 << " = " << tracker.getBGroupExprs()[index0]->evaluate() << "\n";
            // std::cout << "Destination: " << addr1 << " = " << expr->evaluate() << "\n";
            exprMap[addr1] = std::move(expr);
          }
        } else if (row0.getType().getGroup() == 1) {
          assert(row1.getType().getGroup() == 0);
          auto cExpr = getRowIndex(row0) == 0 ? cExpr0.get() : cExpr1.get();
          // std::cout << "Source: " << addr0 << " = " << cExpr->evaluate() << "\n";
          auto index1 = getRowIndex(row1);
          tracker.executeAAP(cExpr, index1);
          // std::cout << "Destination: " << addr1 << " = " << tracker.getBGroupExprs()[index1]->evaluate() << "\n";
        } else {
          assert(row0.getType().getGroup() == 2);
          assert(exprMap.contains(addr0));
          auto src = exprMap[addr0]->clone();
          // std::cout << "Source: " << addr0 << " = " << src->evaluate() << "\n";
          if (row1.getType().getGroup() == 0) {
            auto index1 = getRowIndex(row1);
            tracker.executeAAP(src.get(), index1);
            // std::cout << "Destination: " << addr1 << " = " << tracker.getBGroupExprs()[index1]->evaluate() << "\n";
          } else {
            assert(row1.getType().getGroup() == 2);
            auto expr = tracker.executeAAP(exprMap[addr0].get(), std::nullopt);
            // std::cout << "Destination: " << addr1 << " = " << expr->evaluate() << "\n";
            // std::cout << "Tracking " << addr1 << ": " << expr->evaluate() << "\n";
            exprMap[addr1] = std::move(expr);
          }
        }
        // std::cout << "==============\n";
      }
    });
  });

  std::ofstream trace_file(argv[2]);
  if (!trace_file.is_open()) {
    std::cerr << "Failed to open trace file\n";
    return 1;
  }

  for (const auto &line : trace) {
    trace_file << line << '\n';
  }

  std::cout << "Trace was successfully written into: " << argv[2] << "\n";

  return 0;
}
