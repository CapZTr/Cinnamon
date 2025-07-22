#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Conversion/BitsPasses.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"

#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/Bits/Codegen/ProgramParser.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
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
#include <vector>

#include "ambit.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

using namespace mlir;
using namespace mlir::bits;
using namespace mlir::pud;

using MIG = mockturtle::mig_network;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/BitsPasses.h.inc>

namespace {

struct RowAddress {
  int64_t channel;
  int64_t rank;
  int64_t bank;
  int64_t subarray;
  int64_t row;

  bool inSameSubArray(const RowAddress &other) {
    return this->channel == other.channel
        && this->rank == other.rank
        && this->bank == other.bank
        && this->subarray == other.subarray;
  }

  std::string str() {
    return std::format("{} {} {} {} {}",
        channel,
        rank,
        bank,
        subarray,
        row);
  }
};

class AddressAllocator {
public:
  AddressAllocator() = default;

  RowAddress allocate(int64_t numRows) {
    assert(numRows <= MAX_ROW);

    auto subArrayID = checkSpace(numRows);
    auto saLocalID = subArrayID % 64;

    int64_t channel = currentChannel;
    int64_t rank = currentRank;
    int64_t bank = currentBank;
    int64_t sa = currentSubarray;
    int64_t row = currentRow;
    if (saLocalID == currentBank) {
      currentRow += numRows;
      row = currentRow;
    } else if (subArrayID == -1) {
      currentRow = 0;
      ++currentSubarray;
      if (currentSubarray > MAX_SUBARRAY) {
        currentSubarray = 0;
        ++currentBank;
        if (currentBank > MAX_BANK) {
          currentBank = 0;
          ++currentRank;
          if (currentRank > MAX_RANK) {
            currentRank = 0;
            ++currentChannel;
            if (currentChannel > MAX_CHANNEL) {
              // loops
              // llvm::report_fatal_error(
              //     "AddressAllocator: DRAM address space exhausted");
            }
          }
        }
      }
      channel = currentChannel;
      rank = currentRank;
      bank = currentRank;
      sa = currentSubarray;
      row = currentRow;
    } else {
      int64_t channelSANum = 2 * 8 * 64;
      channel = subArrayID < channelSANum ? 0 : 1;

      auto saChannelID = subArrayID % channelSANum;
      int64_t rankSANum = 8 * 64;

      rank = saChannelID < rankSANum ? 0 : 1;
      auto saRankID = saChannelID % rankSANum;

      bank = saRankID / 8;

      sa = saLocalID;
      auto space = restSpace.lookup(subArrayID);
      row = MAX_ROW - space;
    }

    RowAddress addr = { 
        channel, 
        rank, 
        bank, 
        sa, 
        row};

    int64_t saToUpdate = channel * 2 + rank * 2 + bank * 8 + sa;
    int64_t rest = MAX_ROW - row - numRows;
    restSpace[saToUpdate] = rest;

    return addr;
  }

  RowAddress getRowFromOffset(const RowAddress &base, const int64_t offset) {
    auto rowID = base.row + offset;
    assert(rowID <= MAX_ROW);
    return {base.channel, base.rank, base.bank, base.subarray, rowID};
  }

private:
  const int64_t MAX_CHANNEL = 1;
  const int64_t MAX_RANK = 1;
  const int64_t MAX_BANK = 7;
  const int64_t MAX_SUBARRAY = 63;
  const int64_t MAX_ROW = 1005;
  const int64_t MAX_COLUMN = 8192;

  int64_t currentChannel = 0;
  int64_t currentRank = 0;
  int64_t currentBank = 0;
  int64_t currentSubarray = 0;
  int64_t currentRow = 0;

  DenseMap<int64_t, int64_t> restSpace;

  int checkSpace(int64_t numRows) const {
    int subArrayID = -1;
    for (const auto &pair : restSpace) {
      if (numRows <= pair.second)
        subArrayID = pair.first;
    }
    return subArrayID;
  }
};

struct GlobalAddressAllocator {
  static AddressAllocator &get() {
    static AddressAllocator allocator;
    return allocator;
  }
};

// struct InputCache {
//   static llvm::DenseMap<Value, Value> &get() {
//     static llvm::DenseMap<Value, Value> cache;
//     return cache;
//   }
// };

// struct SliceCache {
//   static llvm::DenseMap<Operation *, Value> &get() {
//     static llvm::DenseMap<Operation *, Value> cache;
//     return cache;
//   }
// };

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
    auto inputSlices = builder.getInputSlices();

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
    // std::cout << "Generated program:\n" << program_str.str() << "\n";

    // NetworkBuilder::debugPrint(optimized);

    // std::cout << " ===== Optimized Network ===== \n";
    // mockturtle::write_dot(optimized, std::cout);


    // =========================================================================
    // ========== Program Parsing ==========
    // =========================================================================

    ProgramParser parser(program_str.str());
    if (failed(parser.parse())) {
      signalPassFailure();
    }
    std::vector<Instruction> program = parser.getProgram();
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

    SmallVector<TypedValue<RowType>> outputs;

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
    // ========== Data rows Allocation ==========
    // =========================================================================

    for (auto &inst : program) {
      if (inst.type == Instruction::Type::AAP) {
        auto operand0 = inst.operand0;
        if (operand0.type == AddressType::In) {
          if (!allocated.contains(operand0.str_repr)) {
            auto data = std::get<int>(operand0.data);
            assert(mapping.contains(inputSlices[data]));
            auto mapped = mapping.lookup(inputSlices[data]);
            auto newSlice = cast<TypedValue<SliceType>>(mapped);
            auto addr = allocate(newSlice.getType().getBitWidth());
            allocated[operand0.str_repr] = addr;
            auto firstRow = getOrCreateDRow(addr);
            auto store = opBuilder.create<StoreOp>(loc, newSlice, firstRow);
            firstRows[operand0.str_repr] = firstRow;
          }
        } else if (operand0.type == AddressType::Spill) {
          assert(allocated.contains(operand0.str_repr));
        }

        auto operand1 = *inst.operand1;
        if (operand1.type == AddressType::Out || operand1.type == AddressType::Spill) {
          if (!allocated.contains(operand1.str_repr)) {
            auto addr = allocate(bitWidth);
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
      while (iterIndex < bitWidth) {

        for (auto &inst : program) {

          if (inst.type == Instruction::Type::AP) {
            assert(inst.operand0.type == AddressType::Bitwise);
            auto index = std::get<int>(inst.operand0.data);
            auto ap = opBuilder.create<APOp>(loc, bRows[index]);
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
              if (iterIndex == 0) {
                assert(firstRows.contains(operand0.str_repr));
                addr0 = firstRows.lookup(operand0.str_repr);
              } else {
                addr0 = getOrCreateDRow(allocator.getRowFromOffset(firstRow, iterIndex));
              }
            }

            if (operand1.type == AddressType::Bitwise) {
              auto index = std::get<int>(operand1.data);
              addr1 = bRows[index];
            } else {
              assert(operand1.type == AddressType::Out || operand1.type == AddressType::Spill);
              auto firstRow = allocated[operand1.str_repr];
              auto rowId = firstRow.row;
              if (iterIndex == 0) {
                assert(firstRows.contains(operand1.str_repr));
                addr1 = firstRows.lookup(operand1.str_repr);
              } else {
                addr1 = getOrCreateDRow(allocator.getRowFromOffset(firstRow, iterIndex));
              }
            }

            auto aap = opBuilder.create<AAPOp>(loc, addr0, addr1);

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
            auto ap = opBuilder.create<APOp>(loc, addr0);
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

        auto aap = opBuilder.create<AAPOp>(loc, addr0, addr1);
      }

      opBuilder.setInsertionPointAfter(loop);

    }

    auto sliceType = SliceType::get(ctx, bitWidth, vecLen);
    auto firstRow = outputs[0];

    Value resultSlice = opBuilder.create<LoadOp>(loc, sliceType, firstRow, getOrCreateI64Val(bitWidth));

    Value resultTensor = opBuilder.create<AssembleOp>(loc, tensorType, resultSlice);

    opBuilder.create<func::ReturnOp>(loc, resultTensor);

  }

  RowAddress allocate(int64_t numRows) {
    auto &allocator = GlobalAddressAllocator::get();
    return allocator.allocate(numRows);
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
