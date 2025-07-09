#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsFrontendPasses.h"
#include "cinm-mlir/Conversion/BitsPasses.h"
#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include <algorithm>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SourceMgr.h>
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

const int NUM_ROW_PER_SUBARRAY = 1024;
const std::string DUMMY_DATA = "b186649dd2c40617e1df8669b90acd6389c0e5f8e5c059c5a4ea4f9eb6409eaacf4380666a43bcc792e0d3f2a7b88eca6067d625801408a3df929bb8b4136b68";

std::string intToHex(const int v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

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

  // std::cout << "Successfully ran passes\n";

  std::vector<std::string> trace;
  trace.push_back(">");

  DenseMap<mlir::TypedValue<IntegerType>, int64_t> indices;
  DenseMap<mlir::TypedValue<pud::RowType>, std::string> addrs;
  
  int cycle = 1;

  auto getAddressAsStr = [&indices, &addrs](mlir::TypedValue<pud::RowType> rowAddr)
      -> std::string {
    if (!addrs.contains(rowAddr)) {
      auto op = rowAddr.getDefiningOp();
      assert(isa<pud::GetRowOp>(*op));
      auto getRow = cast<pud::GetRowOp>(*op);
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
      auto addrAsInt = subarray * NUM_ROW_PER_SUBARRAY + row;
      addrs[rowAddr] = intToHex(addrAsInt);
    }
    return addrs.lookup(rowAddr);
  };

  auto getRowNum = [&](mlir::TypedValue<pud::RowType> rowAddr) -> int {
    auto op = rowAddr.getDefiningOp();
    assert(isa<pud::GetRowOp>(*op));
    auto getRow = cast<pud::GetRowOp>(*op);
    auto row = indices.lookup(getRow.getRowID());
    assert(row >= 0 && row <= 15);
    return row < 8 ? 1 
        : row < 12 ? 2
        : 3;
  };

  module->walk([&](func::FuncOp func) {
    func->walk([&](Operation *op) {
      if (auto cnst = dyn_cast<arith::ConstantOp>(*op)) {
        assert(cnst.getType().isInteger(64));
        indices[cast<mlir::TypedValue<IntegerType>>(cnst.getResult())] = cast<IntegerAttr>
            (cnst.getValue()).getValue().getZExtValue();
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
      } else if (auto aap = dyn_cast<pud::AAPOp>(*op)) {
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
        auto maxNum = std::max(rowNum0, rowNum1);
        const std::string opName = maxNum == 1 ? "O"
            : maxNum == 2 ? "ODRA"
            : "OTRA";
        trace.push_back(getTraceLine(cycle, opName, addr0, std::make_optional(addr1)));
        cycle++;
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
