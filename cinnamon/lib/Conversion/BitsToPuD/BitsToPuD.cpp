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
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

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
  int64_t bank;
  int64_t subarray;
  int64_t row;
};

class AddressAllocator {
public:
  AddressAllocator() = default;

  RowAddress allocate(int64_t numRows) {
    if (currentRow + numRows > MAX_ROW) {
      currentRow = 0;
      ++currentSubarray;
      if (currentSubarray > MAX_SUBARRAY) {
        currentSubarray = 0;
        ++currentBank;
        if (currentBank > MAX_BANK)
          llvm::report_fatal_error(
              "AddressAllocator: DRAM address space exhausted");
      }
    }
    RowAddress addr = {currentBank, currentSubarray, currentRow};
    currentRow += numRows;
    return addr;
  }

private:
  const int64_t MAX_BANK = 15;
  const int64_t MAX_SUBARRAY = 31;
  const int64_t MAX_ROW = 1005;
  int64_t currentBank = 0;
  int64_t currentSubarray = 0;
  int64_t currentRow = 0;
};

struct GlobalAddressAllocator {
  static AddressAllocator &get() {
    static AddressAllocator allocator;
    return allocator;
  }
};

struct InputCache {
  static llvm::DenseMap<Value, Value> &get() {
    static llvm::DenseMap<Value, Value> cache;
    return cache;
  }
};

struct SliceCache {
  static llvm::DenseMap<Operation *, Value> &get() {
    static llvm::DenseMap<Operation *, Value> cache;
    return cache;
  }
};

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
    std::cout << "Generated program:\n" << program_str.str() << "\n";

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
    std::cout << " ===== Parsed " << program.size() << " instructions =====" << "\n";


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

    llvm::StringMap<TypedValue<DataRowAddressType>> allocated;

    SmallVector<TypedValue<DataRowAddressType>> outputs;

    for (auto &inst : program) {
      if (inst.type == Instruction::Type::AAP) {
        auto operand0 = inst.operand0;
        if (operand0.type == AddressType::In) {
          if (!allocated.contains(operand0.str_repr)) {
            auto data = std::get<int>(operand0.data);
            assert(mapping.contains(inputSlices[data]));
            auto mapped = mapping.lookup(inputSlices[data]);
            auto newSlice = cast<TypedValue<SliceType>>(mapped);
            auto addr = allocate(bitWidth);
            auto addrType = DataRowAddressType::get(ctx, addr.bank, addr.subarray, addr.row);
            Value firstRow = opBuilder.create<StoreOp>(loc, addrType, newSlice);
            allocated[operand0.str_repr] = cast<TypedValue<DataRowAddressType>>(firstRow);
          }
        } else if (operand0.type == AddressType::Spill) {
          assert(allocated.contains(operand0.str_repr));
        }

        auto operand1 = *inst.operand1;
        if (operand1.type == AddressType::Out || operand1.type == AddressType::Spill) {
          if (!allocated.contains(operand1.str_repr)) {
            auto addr = allocate(bitWidth);
            auto addrType = DataRowAddressType::get(ctx, addr.bank, addr.subarray, addr.row);
            Value firstRow = opBuilder.create<GetRowOp>(loc, addrType);
            allocated[operand1.str_repr] = cast<TypedValue<DataRowAddressType>>(firstRow);
            if (operand1.type == AddressType::Out)
              outputs.push_back(cast<TypedValue<DataRowAddressType>>(firstRow));
          }
        }
      }
    }

    int64_t iterIndex = 0;
    int64_t currentBank = 0;
    int64_t currentSubarray = 0;
    SmallVector<SmallVector<DenseMap<int64_t, Value>>> bitwiseAddrs;
    SmallVector<SmallVector<DenseMap<int, Value>>> controlAddrs;
    SmallVector<SmallVector<DenseMap<int64_t, Value>>> dataAddrs;

    auto getBitwiseAddress = [&ctx, &loc, &opBuilder, &bitwiseAddrs, &currentBank, &currentSubarray]
        (int index) -> TypedValue<BitwiseRowAddressType> {
      Value bAddr;

      if (bitwiseAddrs.size() <= static_cast<size_t>(currentBank)) {
        bitwiseAddrs.push_back(SmallVector<DenseMap<int64_t, Value>>());
      }

      auto &subarrays = bitwiseAddrs[currentBank];
      if (subarrays.size() <= static_cast<size_t>(currentSubarray)) {
        subarrays.push_back(DenseMap<int64_t, Value>());
      }

      auto &addrs = subarrays[currentSubarray];
      if (!addrs.contains(index)) {
        auto addrType = BitwiseRowAddressType::get(ctx, currentBank, currentSubarray, index);
        Value addr = opBuilder.create<GetRowOp>(loc, addrType);
        addrs[index] = addr;
      }

      bAddr = addrs.lookup(index);
      return cast<TypedValue<BitwiseRowAddressType>>(bAddr);
    };

    auto getControlAddress = [&ctx, &loc, &opBuilder, &controlAddrs, &currentBank, &currentSubarray]
        (int val) -> TypedValue<ControlRowAddressType> {
      Value cAddr;

      if (controlAddrs.size() <= static_cast<size_t>(currentBank)) {
        controlAddrs.push_back(SmallVector<DenseMap<int, Value>>());
      }

      auto &subarrays = controlAddrs[currentBank];
      if (subarrays.size() <= static_cast<size_t>(currentSubarray)) {
        subarrays.push_back(DenseMap<int, Value>());
      }

      auto &addrs = subarrays[currentSubarray];
      if (!addrs.contains(val)) {
        bool v = val == 1 ? true : false;
        auto addrType = ControlRowAddressType::get(ctx, currentBank, currentSubarray, v);
        Value addr = opBuilder.create<GetRowOp>(loc, addrType);
        addrs[val] = addr;
      }

      cAddr = addrs.lookup(val);
      return cast<TypedValue<ControlRowAddressType>>(cAddr);
    };

    auto getDataAddress = [&ctx, &loc, &opBuilder, &dataAddrs, &currentBank, &currentSubarray]
        (int64_t index) -> TypedValue<DataRowAddressType> {
      Value dAddr;

      if (dataAddrs.size() <= static_cast<size_t>(currentBank)) {
        dataAddrs.push_back(SmallVector<DenseMap<int64_t, Value>>());
      }

      auto &subarrays = dataAddrs[currentBank];
      if (subarrays.size() <= static_cast<size_t>(currentSubarray)) {
        subarrays.push_back(DenseMap<int64_t, Value>());
      }

      auto &addrs = subarrays[currentSubarray];
      if (!addrs.contains(index)) {
        auto addrType = DataRowAddressType::get(ctx, currentBank, currentSubarray, index);
        Value addr = opBuilder.create<GetRowOp>(loc, addrType);
        addrs[index] = addr;
      }

      dAddr = addrs.lookup(index);
      return cast<TypedValue<DataRowAddressType>>(dAddr);
    };

    while (iterIndex < bitWidth) {
      for (auto &inst : program) {
        if (inst.type == Instruction::Type::AP) {
          assert(inst.operand0.type == AddressType::Bitwise);
          auto index = std::get<int>(inst.operand0.data);
          auto addr = getBitwiseAddress(index);
          auto ap = opBuilder.create<APOp>(loc, addr);

        } else {
          assert(inst.type == Instruction::Type::AAP);

          Value addr0, addr1;

          auto operand0 = inst.operand0;
          if (operand0.type == AddressType::Bitwise) {
            auto index = std::get<int>(operand0.data);
            addr0 = getBitwiseAddress(index);
          } else if (operand0.type == AddressType::Const) {
            auto val = std::get<bool>(operand0.data) ? 1 : 0;
            addr0 = getControlAddress(val);
          } else {
            assert(operand0.type == AddressType::In || operand0.type == AddressType::Spill);
            auto firstRow = allocated[operand0.str_repr];
            currentBank = firstRow.getType().getBankID();
            currentSubarray = firstRow.getType().getSubarrayID();
            auto rowId = firstRow.getType().getRowID();
            if (iterIndex == 0) {
              addr0 = firstRow;
            } else {
              addr0 = getDataAddress(iterIndex + rowId);
            }
          }

          auto operand1 = *inst.operand1;
          if (operand1.type == AddressType::Bitwise) {
            auto index = std::get<int>(operand1.data);
            addr1 = getBitwiseAddress(index);
          } else {
            assert(operand1.type == AddressType::Out || operand1.type == AddressType::Spill);
            auto firstRow = allocated[operand1.str_repr];
            currentBank = firstRow.getType().getBankID();
            currentSubarray = firstRow.getType().getSubarrayID();
            auto rowId = firstRow.getType().getRowID();
            if (iterIndex == 0) {
              addr1 = firstRow;
            } else {
              addr1 = getDataAddress(iterIndex + rowId);
            }
          }

          auto aap = opBuilder.create<AAPOp>(loc, addr0, addr1);
        }
      }

      iterIndex++;

    }

    // auto tensorType = RankedTensorType::get({64}, opBuilder.getI32Type());
    // Value emptyTensor = opBuilder.create<tensor::EmptyOp>(
    //     loc,
    //     tensorType,
    //     ArrayRef<Value>{}
    // );

    auto sliceType = SliceType::get(ctx, bitWidth, vecLen);
    auto firstRow = outputs[0];

    Value resultSlice = opBuilder.create<LoadOp>(loc, sliceType, firstRow);

    Value resultTensor = opBuilder.create<AssembleOp>(loc, tensorType, resultSlice);

    opBuilder.create<func::ReturnOp>(loc, resultTensor);

  }

  RowAddress allocate(int64_t numRows) {
    auto &allocator = GlobalAddressAllocator::get();
    return allocator.allocate(numRows);
  }

};

} // namespace

std::unique_ptr<Pass> mlir::bits::createConvertBitsToPuDPass() {
  return std::make_unique<ConvertBitsToPuD>();
}
