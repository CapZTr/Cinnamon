#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/PuD/Codegen/AddressAllocator.h"
#include "cinm-mlir/Dialect/PuD/Codegen/ProgramParser.h"

#include <cassert>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <format>
#include <iostream>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/write_dot.hpp>
#include <optional>
#include <ostream>
#include <vector>

#include "ambit.h"

using namespace mlir;
using namespace mlir::bits;
using namespace mlir::pud;

using MIG = mockturtle::mig_network;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/BitsPasses.h.inc>

namespace {

struct GlobalAddressAllocator {
  static AddressAllocator &get() {
    static AddressAllocator allocator;
    return allocator;
  }
};


// =========================================================================
// ========== Pass Structure ==========
// =========================================================================

struct ConvertBitsToPuD
    : public ConvertBitsToPuDBase<ConvertBitsToPuD> {
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // =========================================================================
    // ========== Network Generation ==========
    // =========================================================================

    NetworkBuilder builder(func->getParentOfType<ModuleOp>());
    if (failed(builder.build())) {
      signalPassFailure();
    }
    auto mig = builder.getNetwork();
    auto ntkInputs = builder.getInputSlices();
    const int inputNum = ntkInputs.size();
    auto carryMap = builder.getCarryMap();
    const int carryNum = carryMap.size();
    // auto ntkOutputs = builder.getOutputSlices();

    // NetworkBuilder::debugPrint(mig);

    // std::cout << " ===== Generated Network ===== \n";
    // mockturtle::write_dot(mig, std::cout);

    // =========================================================================
    // ========== Lime Optimization ==========
    // =========================================================================

    const auto settings = ambit_compiler_settings{
        .print_program = false,
        .verbose = false,
        .preoptimize = true,
        .rewrite = false,
    };

    ProgramString program_str;
    auto [optimized, result] = ambit_rewrite(settings, mig, program_str);
    std::cout << "\nGenerated program:\n" << program_str.str() << "\n";

    // NetworkBuilder::debugPrint(optimized);

    // std::cout << "\n ===== Optimized Network ===== \n";
    // mockturtle::write_dot(optimized, std::cout);


    // =========================================================================
    // ========== Program Parsing ==========
    // =========================================================================

    ProgramParser parser(program_str.str());
    if (failed(parser.parse())) {
      signalPassFailure();
    }
    std::vector<Instruction> program = parser.getProgram();
    parser.printProgram();
    // std::cout << " ===== Parsed " << program.size() << " instructions =====" << "\n";


    // =========================================================================
    // ========== IR (PuD Dialect) Building ==========
    // =========================================================================

    Location loc = func.getLoc();
    Block &oldEntry = func.getBody().front();

    SmallVector<Operation *> toKeep;
    for (Operation &op : oldEntry) {
      if (isa<TransposeOp>(op))
        toKeep.push_back(&op);
    }

    auto transpose = cast<TransposeOp>(toKeep[0]);
    auto bitWidth = transpose.getOutput().getType().getBitWidth();
    auto vecLen = transpose.getOutput().getType().getVectorLength();
    auto tensorType = cast<RankedTensorType>(transpose.getInput().getType());

    Block *newEntry = new Block();
    func.getBody().push_back(newEntry);

    auto inputTypes = func.getFunctionType().getInputs();
    SmallVector<Location> argLocs(inputTypes.size(), loc);
    newEntry->addArguments(inputTypes, argLocs);

    OpBuilder opBuilder(newEntry, newEntry->begin());
    auto ctx = opBuilder.getContext();

    // auto resultSliceType = SliceType::get(ctx, bitWidth, vecLen);

    Type i64Type = opBuilder.getIntegerType(64);

    IRMapping mapping;
    auto oldArgs = oldEntry.getArguments();
    auto newArgs = newEntry->getArguments();
    for (auto [oldArg, newArg] : zip(oldArgs, newArgs)) {
      mapping.map(oldArg, newArg);
    }

    for (Operation *oldOp : toKeep) {
      Operation *newOp = opBuilder.clone(*oldOp, mapping);
      assert(isa<TransposeOp>(*oldOp) && isa<TransposeOp>(*newOp));
      auto oldTranspose = cast<TransposeOp>(oldOp);
      auto newTranspose = cast<TransposeOp>(newOp);
      mapping.map(oldTranspose.getOutput(), newTranspose.getOutput());
    }

    oldEntry.dropAllReferences();
    oldEntry.erase();

    DenseMap<int64_t, Value> i64Vals;
    auto getOrCreateI64Val = [&opBuilder, &loc, &i64Type, &i64Vals](int64_t num) -> Value {
      if (!i64Vals.contains(num)) {
        Value val = opBuilder.create<arith::ConstantOp>(
            loc, i64Type, opBuilder.getI64IntegerAttr(num));
        i64Vals[num] = val;
      }
      return i64Vals.lookup(num);
    };

    // =========================================================================
    // ========== Address Allocation ==========
    // =========================================================================

    auto &allocator = GlobalAddressAllocator::get();
    
    llvm::StringMap<RowAddress> allocated;
    llvm::StringMap<TypedValue<RowType>> firstRows;
    llvm::StringMap<TypedValue<RowType>> rowMap;
    llvm::StringMap<TypedValue<RowType>> carryRows;

    SmallVector<TypedValue<RowType>, 2> outputs;
    SmallVector<SmallVector<TypedValue<SliceType>>, 2> outputSlices;
    for (int i = 0; i < 2; ++i) {
      SmallVector<TypedValue<SliceType>> slices;
      outputSlices.push_back(slices);
    }

    auto bRowType = RowType::get(ctx, 0);
    auto cRowType = RowType::get(ctx, 1);
    auto dRowType = RowType::get(ctx, 2);

    auto getOrCreateDRow = [&loc, &opBuilder, &dRowType, &rowMap, &getOrCreateI64Val]
        (RowAddress addr) -> TypedValue<RowType> {
      auto addrStr = addr.str();
      if (!rowMap.contains(addrStr)) {
        Value rowVal = opBuilder.create<GetRowOp>(
            loc,
            dRowType,
            getOrCreateI64Val(addr.channel),
            getOrCreateI64Val(addr.rank),
            getOrCreateI64Val(addr.bank),
            getOrCreateI64Val(addr.subarray),
            getOrCreateI64Val(addr.row)
        );
        rowMap[addrStr] = cast<TypedValue<RowType>>(rowVal);
      }
      return rowMap.lookup(addrStr);
    };

    const Value zero = getOrCreateI64Val(0);

    // =========================================================================
    // ========== Bitwise & Control Rows Generation ==========
    // TODO: This is a workaround.
    //       Currently we use B and C group rows from subarry (0, 0, 0, 0) only.
    //       We need an algorithm to find the optimality with lowest latency of
    //       inter-subarray row clone.
    // =========================================================================

    SmallVector<TypedValue<RowType>, 16> bRows;
    for (int i = 0; i < 16; ++i) {
      Value bRow = opBuilder.create<GetRowOp>(
          loc,
          bRowType,
          zero,
          zero,
          zero,
          zero,
          getOrCreateI64Val(i)
      );
      bRows.push_back(cast<TypedValue<RowType>>(bRow));
    }

    const TypedValue<RowType> c0 = opBuilder.create<GetRowOp>(
        loc,
        cRowType,
        zero,
        zero,
        zero,
        zero,
        zero
    );
    const TypedValue<RowType> c1 = opBuilder.create<GetRowOp>(
        loc,
        cRowType,
        zero,
        zero,
        zero,
        zero,
        getOrCreateI64Val(1)
    );

    // =========================================================================
    // ========== Data Rows Allocation ==========
    // =========================================================================

    auto maxColNum = allocator.getMaxColumnNum();
    int lastSliceVecLen = vecLen % maxColNum;
    if (lastSliceVecLen == 0)
      lastSliceVecLen = maxColNum;
    int round = (vecLen - 1) / maxColNum + 1;
    Value maxColVal = getOrCreateI64Val(maxColNum);
    auto standardSliceType = SliceType::get(ctx, bitWidth, maxColNum);
    
    SmallVector<SmallVector<TypedValue<SliceType>>> slicesToStore;
    for (auto &oldSlice : ntkInputs) {
      assert(mapping.contains(oldSlice));
      auto newSlice = cast<TypedValue<SliceType>>(mapping.lookup(oldSlice));
      SmallVector<TypedValue<SliceType>> v;
      v.push_back(newSlice);
      for (int roundIdx = 0; roundIdx < round - 1; ++roundIdx) {
        auto toSplit = v[roundIdx];
        auto secondColNum = toSplit.getType().getVectorLength() - maxColNum;
        if (roundIdx + 2 == round)
          assert(secondColNum <= maxColNum);
        auto secondType = SliceType::get(ctx, bitWidth, secondColNum);
        auto splitOp = opBuilder.create<SplitSliceVerticallyOp>(
            loc,
            standardSliceType,
            secondType,
            toSplit,
            maxColVal
        );
        v[roundIdx] = splitOp.getFirst();
        v.push_back(splitOp.getSecond());
      }
      slicesToStore.push_back(v);
    }

    for (int roundIdx = 0; roundIdx < round; ++roundIdx) {
      allocator.reset();
      StringSet<> stored;

      for (auto &inst : program) {
        if (inst.type == Instruction::Type::AAP) {
          auto operand0 = inst.operand0;
          if (operand0.type == AddressType::In) {
            const auto inputIdx = std::get<int>(operand0.data);
            if (inputIdx < inputNum) {
              if (!allocated.contains(operand0.str_repr)) {
                assert(roundIdx == 0);
                // allocator.printStatus();
                auto addr = allocator.allocate(bitWidth);
                // std::cout << operand0.str_repr << ": " << addr.str() << "\n";
                // allocator.printStatus();
                allocated[operand0.str_repr] = addr;
                auto firstRow = getOrCreateDRow(addr);
                firstRows[operand0.str_repr] = firstRow;
              }
              if (!stored.contains(operand0.str_repr)) {
                auto data = std::get<int>(operand0.data);
                auto toStore = slicesToStore[data][roundIdx];
                assert(firstRows.contains(operand0.str_repr));
                opBuilder.create<StoreOp>(loc, toStore, firstRows.lookup(operand0.str_repr));
                stored.insert(operand0.str_repr);
              }
            } else {
              if (!allocated.contains(operand0.str_repr)) {
                assert(roundIdx == 0);
                // allocator.printStatus();
                auto addr = allocator.allocate(1);
                // std::cout << operand0.str_repr << ": " << addr.str() << "\n";
                // allocator.printStatus();
                allocated[operand0.str_repr] = addr;
                auto cinRow = getOrCreateDRow(addr);
                assert(!carryRows.contains(operand0.str_repr));
                carryRows.try_emplace(operand0.str_repr, cinRow);
              }
            }
          } else if (operand0.type == AddressType::Spill) {
            assert(allocated.contains(operand0.str_repr));
          }

          auto operand1 = *inst.operand1;
          if (operand1.type == AddressType::Out || operand1.type == AddressType::Spill) {
            const auto index = std::get<int>(operand1.data);
            if (operand1.type == AddressType::Out && index < carryNum) {
              if (!allocated.contains(operand1.str_repr)) {
                assert(roundIdx == 0);
                // allocator.printStatus();
                auto addr = allocator.allocate(1);
                // std::cout << operand1.str_repr << ": " << addr.str() << "\n";
                // allocator.printStatus();
                allocated[operand1.str_repr] = addr;
                auto coutRow = getOrCreateDRow(addr);
                assert(!carryRows.contains(operand1.str_repr));
                carryRows.try_emplace(operand1.str_repr, coutRow);
              }
              continue;
            }
            if (!allocated.contains(operand1.str_repr)) {
              assert(roundIdx == 0);
              // allocator.printStatus();
              auto addr = allocator.allocate(bitWidth);
              // std::cout << operand1.str_repr << ": " << addr.str() << "\n";
              // allocator.printStatus();
              allocated[operand1.str_repr] = addr;
              auto firstRow = getOrCreateDRow(addr);
              firstRows[operand1.str_repr] = firstRow;
              if (operand1.type == AddressType::Out)
                outputs.push_back(firstRow);
            }
          } else if (operand1.type == AddressType::Bitwise) {
          }
        } else {
          assert(inst.type == Instruction::Type::AP);
        }
      }

      if (this->unroll) {

        int64_t iterIndex = 0;
        int64_t addrOffset = bitWidth;
        while (iterIndex < bitWidth) {
          addrOffset--;
          llvm::StringSet<> refreshedCin;
          for (auto &inst : program) {

            if (inst.type == Instruction::Type::AP) {
              assert(inst.operand0.type == AddressType::Bitwise);
              auto index = std::get<int>(inst.operand0.data);
              opBuilder.create<APOp>(loc, bRows[index]);
            } else {
              assert(inst.type == Instruction::Type::AAP);

              TypedValue<RowType> addr0, addr1;

              auto operand0 = inst.operand0;
              auto operand1 = *inst.operand1;

              if (operand0.type == AddressType::Bitwise) {
                auto index = std::get<int>(operand0.data);
                addr0 = bRows[index];
              } else if (operand0.type == AddressType::Const) {
                addr0 = std::get<bool>(operand0.data) ? c1 : c0;
              } else {
                assert(operand0.type == AddressType::In || operand0.type == AddressType::Spill);
                auto firstRow = allocated[operand0.str_repr];
                if (carryRows.contains(operand0.str_repr)) {
                  if (iterIndex == 0) {
                    addr0 = c0;
                  } else {
                    addr0 = carryRows.lookup(operand0.str_repr);
                    if (!refreshedCin.contains(operand0.str_repr)) {
                      const auto cinIdx = std::get<int>(operand0.data);
                      const auto coutIdx = carryMap.lookup(cinIdx);
                      auto carry = carryRows.lookup(std::format("O{}", coutIdx));
                      opBuilder.create<AAPOp>(loc, carry, addr0);
                      refreshedCin.insert(operand0.str_repr);
                    }
                  }
                } else {
                  if (addrOffset == 0) {
                    assert(firstRows.contains(operand0.str_repr));
                    addr0 = firstRows.lookup(operand0.str_repr);
                  } else {
                    addr0 = getOrCreateDRow(allocator.getRowFromOffset(firstRow, addrOffset));
                  }
                }
              }

              if (operand1.type == AddressType::Bitwise) {
                auto index = std::get<int>(operand1.data);
                addr1 = bRows[index];
              } else {
                assert(operand1.type == AddressType::Out || operand1.type == AddressType::Spill);
                if (carryRows.contains(operand1.str_repr)) {
                  assert(operand1.type == AddressType::Out);
                  const auto outputIdx = std::get<int>(operand1.data);
                  assert(outputIdx < carryNum);
                  addr1 = carryRows.lookup(operand1.str_repr);
                } else {
                  auto firstRow = allocated[operand1.str_repr];
                  if (addrOffset == 0) {
                    assert(firstRows.contains(operand1.str_repr));
                    addr1 = firstRows.lookup(operand1.str_repr);
                  } else {
                    addr1 = getOrCreateDRow(allocator.getRowFromOffset(firstRow, addrOffset));
                  }
                }
              }

              opBuilder.create<AAPOp>(loc, addr0, addr1);

            }
          }

          iterIndex++;

        }

      } else {

        auto loop = opBuilder.create<affine::AffineForOp>(loc, 0, bitWidth, 1);

        opBuilder.setInsertionPointToStart(loop.getBody());

        Value iterIndex = opBuilder.create<arith::IndexCastOp>(loc, i64Type, loop.getInductionVar());

        for (auto &inst : program) {
          auto operand0 = inst.operand0;
          TypedValue<RowType> addr0;
          if (operand0.type == AddressType::Bitwise) {
            auto index = std::get<int>(operand0.data);
            addr0 = bRows[index];
            if (inst.type == Instruction::Type::AP) {
              opBuilder.create<APOp>(loc, addr0);
              continue;
            }
          } else if (operand0.type == AddressType::Const) {
            addr0 = std::get<bool>(operand0.data) ? c1 : c0;
          } else {
            assert(operand0.type == AddressType::In || operand0.type == AddressType::Spill);
            auto row = allocated.lookup(operand0.str_repr);
            Value rowID = opBuilder.create<arith::AddIOp>(loc, i64Type, getOrCreateI64Val(row.row), iterIndex);
            addr0 = opBuilder.create<GetRowOp>(
                loc,
                dRowType,
                getOrCreateI64Val(row.channel),
                getOrCreateI64Val(row.rank),
                getOrCreateI64Val(row.bank),
                getOrCreateI64Val(row.subarray),
                rowID
            );
          }

          assert(inst.type == Instruction::Type::AAP);

          auto operand1 = *inst.operand1;
          TypedValue<RowType> addr1;

          if (operand1.type == AddressType::Bitwise) {
            auto index = std::get<int>(operand1.data);
            addr1 = bRows[index];
          } else {
            assert(operand1.type == AddressType::Out || operand1.type == AddressType::Spill);
            auto row = allocated.lookup(operand1.str_repr);
            Value rowID = opBuilder.create<arith::AddIOp>(loc, i64Type, getOrCreateI64Val(row.row), iterIndex);
            addr1 = opBuilder.create<GetRowOp>(
                loc,
                dRowType,
                getOrCreateI64Val(row.channel),
                getOrCreateI64Val(row.rank),
                getOrCreateI64Val(row.bank),
                getOrCreateI64Val(row.subarray),
                rowID
            );
          }

          opBuilder.create<AAPOp>(loc, addr0, addr1);
        }

        opBuilder.setInsertionPointAfter(loop);

      }

      auto firstRow = outputs[0];
      SliceType sType = standardSliceType;
      if (roundIdx + 1 == round) {
        sType = SliceType::get(ctx, bitWidth, lastSliceVecLen);
      }
      TypedValue<SliceType> slice = opBuilder.create<LoadOp>(loc, sType, firstRow,getOrCreateI64Val(bitWidth));
      outputSlices[0].push_back(slice);

    }

    auto resultSlice = outputSlices[0][0];
    if (round > 1) {
      for (int i = 1; i < round; ++i) {
        auto second = outputSlices[0][i];
        auto vecLen = resultSlice.getType().getVectorLength() + second.getType().getVectorLength();
        auto sType = SliceType::get(ctx, bitWidth, vecLen);
        resultSlice = opBuilder.create<MergeSliceVerticallyOp>(
            loc,
            sType,
            resultSlice,
            second
        );
      }
    }

    Value resultTensor = opBuilder.create<AssembleOp>(loc, tensorType, resultSlice);

    opBuilder.create<func::ReturnOp>(loc, resultTensor);

  }

  void setUnroll(bool doUnroll) {
    this->unroll = doUnroll;
  }

};

} // namespace

std::unique_ptr<Pass> mlir::bits::createConvertBitsToPuDPass() {
  return std::make_unique<ConvertBitsToPuD>();
}

std::unique_ptr<Pass> mlir::bits::createConvertBitsToPuDPass(bool doUnroll) {
  auto pass = std::make_unique<ConvertBitsToPuD>();
  pass->setUnroll(doUnroll);
  return pass;
}
