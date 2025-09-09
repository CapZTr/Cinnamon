#include "cinm-mlir/Conversion/ArithToBits/ArithToBits.h"
#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDBase.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "cinm-mlir/Dialect/PuD/Codegen/OperandTracker.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
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
const std::string DUMMY_DATA = "b186649dd2c40617e1df8669b90acd6389c0e5f8e5c059c"
    "5a4ea4f9eb6409eaacf4380666a43bcc792e0d3f2a7b88eca6067d625801408a3df929bb8b"
    "4136b68";
// For simplification, we set lenth of each vector to 4 when evaluating
// functional correctness.
const size_t VEC_LEN = 4;

std::string intToHex(const int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

// Evaluate Functional Correctness
const auto cExpr0 = std::make_unique<ConstantExpr>(false);
const auto cExpr1 = std::make_unique<ConstantExpr>(true);

llvm::DenseMap<Value, llvm::SmallVector<llvm::APInt>> testIntValMap;
llvm::DenseMap<Value, llvm::SmallVector<llvm::APFloat>> testFloatValMap;
llvm::SetVector<Value> inputs;

unsigned bitWidth = 0;

std::mt19937_64 &rng() {
  thread_local std::mt19937_64 eng{std::random_device{}()};
  return eng;
}

llvm::APInt createRandomAPInt(unsigned N) {
  assert(N > 0);

  const unsigned words = (N + 63) / 64;
  llvm::SmallVector<uint64_t, 4> data(words);

  std::uniform_int_distribution<uint64_t> dist(
      0, std::numeric_limits<uint32_t>::max());
  for (unsigned i = 0; i < words; ++i)
    data[i] = dist(rng());

  const unsigned extra = words * 64 - N;
  if (extra) {
    data.back() &= (extra == 64 ? 0ull : (~0ull >> extra));
  }

  return llvm::APInt(N, words, data.data());
}

llvm::APFloat createAPFloat(unsigned bitWidth, std::optional<APInt> literal) {
  if (literal.has_value()) {
    assert(bitWidth == (*literal).getBitWidth());
  }
  const llvm::fltSemantics *sem = nullptr;
  switch (bitWidth) {
    case 16:
      sem = &llvm::APFloat::IEEEhalf();
      break;
    case 32:
      sem = &llvm::APFloat::IEEEsingle();
      break;
    case 64:
      sem = &llvm::APFloat::IEEEdouble();
      break;
    case 128:
      sem = &llvm::APFloat::IEEEquad();
      break;
    default:
      assert(false && "Only 16/32/64/128-bit APFloat are supported");
  }

  llvm::APInt bits = literal.has_value() ?
      *literal : createRandomAPInt(bitWidth);

  return llvm::APFloat(*sem, bits);
}

llvm::APFloat getOrCreateFloatTestVal(Value v, size_t i) {
  if (v.getDefiningOp() == nullptr) {
    inputs.insert(v);
  }
  assert(i < VEC_LEN);
  assert(llvm::isa<RankedTensorType>(v.getType()));
  auto t = cast<RankedTensorType>(v.getType());
  assert(t.getRank() == 1);
  assert(llvm::isa<FloatType>(t.getElementType()));
  auto elemBitWidth = t.getElementTypeBitWidth();
  if (bitWidth == 0) {
    bitWidth = elemBitWidth;
  } else {
    assert(bitWidth == elemBitWidth);
  }
  if (!testFloatValMap.contains(v)) {
    llvm::SmallVector<llvm::APFloat> vec;
    vec.push_back(createAPFloat(elemBitWidth, std::nullopt));
    testFloatValMap[v] = vec;
  } else if (testFloatValMap[v].size() == i) {
    llvm::APFloat input = createAPFloat(elemBitWidth, std::nullopt);
    testFloatValMap[v].push_back(input);
  }
  return testFloatValMap.lookup(v)[i];
}

llvm::APInt getOrCreateIntTestVal(Value v, size_t i) {
  if (v.getDefiningOp() == nullptr) {
    inputs.insert(v);
  }
  assert(i < VEC_LEN);
  assert(llvm::isa<RankedTensorType>(v.getType()));
  auto t = cast<RankedTensorType>(v.getType());
  assert(t.getRank() == 1);
  assert(llvm::isa<IntegerType>(t.getElementType()));
  auto elemBitWidth = t.getElementTypeBitWidth();
  if (bitWidth == 0) {
    bitWidth = elemBitWidth;
  } else {
    assert(bitWidth == elemBitWidth);
  }
  if (!testIntValMap.contains(v)) {
    llvm::SmallVector<llvm::APInt> vec;
    vec.push_back(createRandomAPInt(elemBitWidth));
    testIntValMap[v] = vec;
  } else if (testIntValMap[v].size() == i) {
    llvm::APInt input = createRandomAPInt(elemBitWidth);
    testIntValMap[v].push_back(input);
  }
  return testIntValMap.lookup(v)[i];
}

void doAddition(Value lhs, Value rhs, Value sum) {
  assert(!testIntValMap.contains(sum));
  llvm::SmallVector<llvm::APInt> resVec;
  for (size_t i = 0; i < VEC_LEN; ++i) {
    llvm::APInt res = getOrCreateIntTestVal(lhs, i) +
        getOrCreateIntTestVal(rhs, i);
    resVec.push_back(res);
  }
  testIntValMap[sum] = resVec;
}

llvm::APFloat apfMul(const llvm::APFloat &a, const llvm::APFloat &b) {
  assert(&a.getSemantics() == &b.getSemantics() &&
      "APFloat semantics must match");
  llvm::APFloat r = a;
  r.multiply(b, llvm::APFloat::rmNearestTiesToEven);
  return r;
}

void doMultiplication(Value lhs, Value rhs, Value product) {
  assert(!testFloatValMap.contains(product));
  llvm::SmallVector<llvm::APFloat> resVec;
  for (size_t i = 0; i < VEC_LEN; ++i) {
    llvm::APFloat res = apfMul(
        getOrCreateFloatTestVal(lhs, i), getOrCreateFloatTestVal(rhs, i));
    resVec.push_back(res);
  }
  testFloatValMap[product] = resVec;
}

llvm::StringMap<llvm::SmallVector<std::unique_ptr<OperandExpr>>> exprMap;

std::string resultIntString(const llvm::SmallVector<llvm::APInt> &resVec) {
  assert(resVec.size() == VEC_LEN);
  std::string resLine = "[";
  for (size_t i = 0; i < VEC_LEN; ++i) {
    llvm::SmallString<64> resStr;
    resVec[i].toString(resStr, 10, true);
    std::string s(resStr.begin(), resStr.end());
    resLine += s;
    resLine += ", ";
  }
  resLine.resize(resLine.size() - 2);
  resLine += "]";
  return resLine;
}

std::string resultFloatString(const llvm::SmallVector<llvm::APFloat> &resVec) {
  assert(resVec.size() == VEC_LEN);
  std::string resLine = "[";
  for (size_t i = 0; i < VEC_LEN; ++i) {
    llvm::SmallString<64> resStr;
    resVec[i].toString(resStr, 4, false);
    std::string s(resStr.begin(), resStr.end());
    resLine += s;
    resLine += ", ";
  }
  resLine.resize(resLine.size() - 2);
  resLine += "]";
  return resLine;
}

void storeAndInitInput(std::string baseAddr, size_t inputIdx, bool isMulF) {
  assert(inputIdx < inputs.size());
  unsigned long long base = std::stoull(baseAddr, nullptr, 0);
  const auto &v = inputs[inputIdx];
  llvm::SmallVector<llvm::APInt> inputLiteralVec;
  if (isMulF) {
    assert(testFloatValMap.contains(v));
    for (const auto &inputFloat : testFloatValMap[v]) {
      inputLiteralVec.push_back(inputFloat.bitcastToAPInt());
    }
  } else {
    assert(testIntValMap.contains(v));
    inputLiteralVec = testIntValMap.lookup(v);
  }
  for (size_t i = 0; i < bitWidth; ++i) {
    auto bitIdx = bitWidth - i - 1;
    llvm::SmallVector<std::unique_ptr<OperandExpr>> exprVec;
    auto addr = intToHex(base + i);
    assert(!exprMap.contains(addr));
    auto name = std::format("I{}_{}", inputIdx, i);
    for (size_t j = 0; j < VEC_LEN; ++j) {
      auto expr = std::make_unique<DataExpr>(name, inputLiteralVec[j][bitIdx]);
      exprVec.push_back(std::move(expr));
    }
    exprMap[addr] = std::move(exprVec);
  }
}

void loadAndEvaluateResult(std::string baseAddr, bool isMulF) {
  unsigned long long base = std::stoull(baseAddr, nullptr, 0);
  llvm::SmallVector<llvm::APInt> outputVec;
  for (size_t i = 0; i < VEC_LEN; ++i) {
    outputVec.push_back(llvm::APInt::getZero(bitWidth));
  }
  for (size_t i = 0; i < bitWidth; ++i) {
    auto bitIdx = bitWidth - i - 1;
    auto addr = intToHex(base + i);
    assert(exprMap.contains(addr));
    const auto &exprVec = exprMap[addr];
    for (size_t j = 0; j < VEC_LEN; ++j) {
      if (exprVec[j]->evaluate()) {
        outputVec[j].setBit(bitIdx);
      }
    }
  }
  if (!isMulF) {
    std::cout << "DRAM result:\n" << resultIntString(outputVec) << "\n\n";
  } else {
    llvm::SmallVector<APFloat> floatVec;
    for (const auto &apInt : outputVec) {
      floatVec.push_back(createAPFloat(bitWidth, apInt));
    }
    std::cout << "DRAM result:\n" << resultFloatString(floatVec) << "\n\n";
  }
}

std::string getTraceLine(const int cycle, const std::string &opName,
    const std::string &addr0, const std::optional<std::string> addr1) {
  std::string line = std::format(
      "{} {} {} {} 0", cycle, opName, addr0, DUMMY_DATA);
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

  std::string cpuRes;
  bool isAddI = false;
  bool isMulF = false;
  func::FuncOp fp = *module->getOps<func::FuncOp>().begin();
  fp.walk([&](Operation *op) {
    if (auto add = dyn_cast<arith::AddIOp>(*op)) {
      if (!isAddI) {
        isAddI = true;
      }
      assert(!isMulF);
      doAddition(add.getLhs(), add.getRhs(), add.getResult());
    } else if (auto mul = dyn_cast<arith::MulFOp>(*op)) {
      if (!isMulF) {
        isMulF = true;
      }
      assert(!isAddI);
      doMultiplication(mul.getLhs(), mul.getRhs(), mul.getResult());
    }
    else if (auto ret = dyn_cast<func::ReturnOp>(*op)) {
      const Value toRet = ret.getOperand(0);
      if (testIntValMap.contains(toRet)) {
        assert(!testFloatValMap.contains(toRet));
        const auto &resVec = testIntValMap[toRet];
        cpuRes = resultIntString(resVec);
      } else {
        assert(testFloatValMap.contains(toRet));
        const auto &resVec = testFloatValMap[toRet];
        cpuRes = resultFloatString(resVec);
      }
    }
  });

  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(bits_frontend::createConvertArithToBitsPass());
  pm.addNestedPass<func::FuncOp>(mlir::bits::createConvertBitsToPuDPass(true));

  if(failed(pm.run(*module))) {
    std::cerr << "Failed to run passes\n";
    return 1;
  }

  std::cout << "Inputs: \n";
  for (const auto &in : inputs) {
    if (isAddI) {
      assert(!isMulF);
      assert(testIntValMap.contains(in));
      std::cout << resultIntString(testIntValMap[in]) << "\n";
    } else {
      assert(isMulF);
      assert(testFloatValMap.contains(in));
      std::cout << resultFloatString(testFloatValMap[in]) << "\n";
    }
  }
  std::cout << "\n";

  std::vector<std::string> trace;
  trace.push_back(">");

  DenseMap<mlir::TypedValue<IntegerType>, int64_t> indices;
  DenseMap<mlir::TypedValue<pud::RowType>, std::string> addrs;
  DenseMap<mlir::TypedValue<pud::RowType>, int64_t> saMap;
  
  int cycle = 1;

  auto getAddressAsStr = [&indices, &addrs, &saMap]
      (mlir::TypedValue<pud::RowType> rowAddr) -> std::string {
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

  llvm::SmallVector<OperandTracker, 16> trackers;
  trackers.resize(VEC_LEN);

  std::string resultAddr;
  module->walk([&](func::FuncOp func) {
    size_t inputIdx = 0;
    func->walk([&](Operation *op) {
      if (auto cnst = dyn_cast<arith::ConstantOp>(*op)) {
        assert(cnst.getType().isInteger(64));
        indices[cast<mlir::TypedValue<IntegerType>>(cnst.getResult())] =
            cast<IntegerAttr>(cnst.getValue()).getValue().getZExtValue();
      } else if (auto store = dyn_cast<pud::StoreOp>(*op)) {
        auto firstRow = store.getFirstRow();
        auto addr = getAddressAsStr(firstRow);
        storeAndInitInput(addr, inputIdx, isMulF);
        inputIdx++;
      } else if (auto load = dyn_cast<pud::LoadOp>(*op)) {
        auto firstRow = load.getFirstRow();
        resultAddr = getAddressAsStr(firstRow);
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
        for (auto &t : trackers) {
          t.executeAP(getRowIndex(row));
        }
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
        if (saMap.lookup(row0) == saMap.lookup(row1)) {
          auto maxNum = std::max(rowNum0, rowNum1);
          const std::string opName = maxNum == 1 ? "O"
              : maxNum == 2 ? "ODRA"
              : "OTRA";
          trace.push_back(
              getTraceLine(cycle, opName, addr0, std::make_optional(addr1)));
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
            for (auto &t : trackers) {
              t.executeAAP(index0, index1);
            }
          } else {
            assert(row1.getType().getGroup() == 2);
            if (!exprMap.contains(addr1)) {
              exprMap[addr1] = llvm::SmallVector<
                  std::unique_ptr<OperandExpr>>();
            }
            for (size_t i = 0; i < VEC_LEN; ++i) {
              auto expr = trackers[i].executeAAP(index0, std::nullopt);
              if (exprMap[addr1].size() <= i) {
                exprMap[addr1].push_back(std::move(expr));
              } else {
                exprMap[addr1][i] = std::move(expr);
              }
            }
          }
        } else if (row0.getType().getGroup() == 1) {
          assert(row1.getType().getGroup() == 0);
          auto cExpr = getRowIndex(row0) == 0 ? cExpr0.get() : cExpr1.get();
          auto index1 = getRowIndex(row1);
          for (auto &t : trackers) {
            t.executeAAP(cExpr, index1);
          }
        } else {
          assert(row0.getType().getGroup() == 2);
          assert(exprMap.contains(addr0));
          if (!exprMap.contains(addr1)) {
            exprMap[addr1] = llvm::SmallVector<std::unique_ptr<OperandExpr>>();
          }
          for (size_t i = 0; i < VEC_LEN; ++i) {
            auto src = exprMap[addr0][i]->clone();
            if (row1.getType().getGroup() == 0) {
              auto index1 = getRowIndex(row1);
              trackers[i].executeAAP(src.get(), index1);
            } else {
              assert(row1.getType().getGroup() == 2);
              auto expr = trackers[i].executeAAP(src.get(), std::nullopt);
              if (exprMap[addr1].size() <= i) {
                exprMap[addr1].push_back(std::move(expr));
              } else {
                exprMap[addr1][i] = std::move(expr);
              }
            }
          }
        }
      }
    });
  });

  std::cout << "CPU result:\n" << cpuRes << "\n\n";
  loadAndEvaluateResult(resultAddr, isMulF);

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
