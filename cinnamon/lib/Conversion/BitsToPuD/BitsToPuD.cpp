#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "cinm-mlir/Dialect/Bits/Codegen/NetworkBuilder.h"
#include "cinm-mlir/Dialect/PuD/Codegen/AddressAllocator.h"
#include "cinm-mlir/Dialect/PuD/Codegen/InstructionMapper.h"
#include "cinm-mlir/Dialect/PuD/Codegen/ProgramParser.h"

#include <iostream>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
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

#include <cassert>
#include <cstdint>
#include <memory>
#include <mockturtle/generators/arithmetic.hpp>
#include <mockturtle/io/write_dot.hpp>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::bits;
using namespace mlir::pud;

using MIG = mockturtle::mig_network;

#define GEN_PASS_CLASSES
#include <cinm-mlir/Conversion/BitsPasses.h.inc>

namespace {

// =============================================================================
// ========================== Pass Structure ===================================
// =============================================================================

struct ConvertBitsToPuD
    : public ConvertBitsToPuDBase<ConvertBitsToPuD> {
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // =========================================================================
    // ======================== Network Generation =============================
    // =========================================================================

    // NetworkBuilder builder(func->getParentOfType<ModuleOp>());
    NetworkBuilder builder(func);
    if (failed(builder.build())) {
      signalPassFailure();
    }
    auto mig = builder.getNetwork();
    if (builder.hasSubgraphs()) {
      // auto subgraphs = builder.getSubgraphNetworks();
      // for (size_t idx = 0; idx < subgraphs.size(); ++idx) {
      //   NetworkBuilder::debugPrint(subgraphs[idx].mig);
      //   mockturtle::write_dot(subgraphs[idx].mig, std::cout);
      // }
      for (const auto &dep : builder.getSubgraphDependencies()) {
        dep.print(llvm::errs());
      }
    } else {
      NetworkBuilder::debugPrint(mig);
    }
    // return;
    auto mulFSignNtk = builder.getMulFSignNtk();
    auto mulFExponentNtk = builder.getMulFExponentNtk();
    const auto isMulF = builder.isMulFNtk();
    auto gtNtk = builder.getGTNtk();
    const auto isMax = builder.isMaxNtk();
    auto ltNtk = builder.getLTNtk();
    const auto isMin = builder.isMinNtk();
    assert((int)isMulF + (int)isMax + (int)isMin <= 1);
    auto ntkInputs = builder.getInputSlices();
    const bool isRed = ntkInputs.size() == 1;
    auto carryMap = builder.getCarryMap();

    // NetworkBuilder::debugPrint(mig);

    // std::cout << " ===== Generated Network ===== \n";
    // mockturtle::write_dot(mig, std::cout);

    // // MulFSignNtk and MulFExponentNtk
    // std::vector<Instruction> program_mulf_sign;
    // std::vector<Instruction> program_minus_bias;
    // if (isMulF) {
    //   ProgramString mulf_sign_str;
    //   auto [optimized_mulf_sign, result_mulf_sign] = ambit_rewrite(
    //       settings, mulFSignNtk, mulf_sign_str);
    //   // std::cout << "\n ===== Optimized Network of MulF Sign-bit ===== \n";
    //   // mockturtle::write_dot(optimized_mulf_sign, std::cout);
    //   ProgramParser parser_mulf_sign(mulf_sign_str.str());
    //   if (failed(parser_mulf_sign.parse())) {
    //     signalPassFailure();
    //   }
    //   program_mulf_sign = parser_mulf_sign.getProgram();
    //   // parser_mulf_sign.printProgram();

    //   ProgramString minus_bias_str;
    //   auto [optimized_minus_bias, result_minus_bias] = ambit_rewrite(
    //       settings, mulFExponentNtk, minus_bias_str);
    //   ProgramParser parser_minus_bias(minus_bias_str.str());
    //   if (failed(parser_minus_bias.parse())) {
    //     signalPassFailure();
    //   }
    //   program_minus_bias = parser_minus_bias.getProgram();
    //   // parser_minus_bias.printProgram();
    // }

    // std::vector<Instruction> program_gt;
    // if (isMax) {
    //   ProgramString gt_str;
    //   auto [optimized_gt, result_gt] = ambit_rewrite(
    //       settings, gtNtk, gt_str);
    //   ProgramParser parser_gt(gt_str.str());
    //   if (failed(parser_gt.parse())) {
    //     signalPassFailure();
    //   }
    //   program_gt = parser_gt.getProgram();
    //   parser_gt.printProgram();
    // }
    // std::vector<Instruction> program_lt;
    // if (isMin) {
    //   ProgramString lt_str;
    //   auto [optimized_lt, result_lt] = ambit_rewrite(
    //       settings, ltNtk, lt_str);
    //   ProgramParser parser_lt(lt_str.str());
    //   if (failed(parser_lt.parse())) {
    //     signalPassFailure();
    //   }
    //   program_lt = parser_lt.getProgram();
    //   parser_lt.printProgram();
    // }


    // =========================================================================
    // ==================== IR (PuD Dialect) Building ==========================
    // =========================================================================

    Location loc = func.getLoc();
    Block &oldEntry = func.getBody().front();

    SmallVector<Operation *> toKeep;
    for (Operation &op : oldEntry) {
      if (isa<TransposeOp>(op))
        toKeep.push_back(&op);
    }

    auto transpose = cast<TransposeOp>(toKeep[0]);
    const auto bitWidth = transpose.getOutput().getType().getBitWidth();
    SliceType exponentBiasT;
    TypedValue<SliceType> exponentBias;
    int exponentBitWidth;
    if (isMulF) {
      switch (bitWidth) {
        case 8:
          exponentBitWidth = 4;
          break;
        case 16:
          exponentBitWidth = 5;
          break;
        case 32:
          exponentBitWidth = 8;
          break;
        case 64:
          exponentBitWidth = 11;
          break;
        case 128:
          exponentBitWidth = 15;
          break;
        default:
          assert(false && "Unsupported floating-point bitwidth");
          break;
      }
    }
    TypedValue<RowType> maskRow;
    const auto vecLen = transpose.getOutput().getType().getVectorLength();
    auto tensorType = cast<RankedTensorType>(
        transpose.getInput().getType());
    if (isRed) {
      auto i1Type = IntegerType::get(tensorType.getContext(), 1);
      tensorType = RankedTensorType::get(
          tensorType.getShape(), i1Type, tensorType.getEncoding());
    }

    DenseMap<Value, int64_t> valueToValueId;
    AssembleOp assembleOp;
    for (Operation &op : oldEntry) {
      if (auto valuesAttr = op.getAttrOfType<ArrayAttr>("bits.value_ids")) {
        for (auto [idx, valueAttr] :
             llvm::enumerate(valuesAttr.getAsRange<IntegerAttr>())) {
          auto valueId = valueAttr.getInt();
          Value result = op.getResult(idx);
          valueToValueId[result] = valueId;
        }
      }
      if (auto assemble = dyn_cast<AssembleOp>(op)) {
        assembleOp = assemble;
      }
    }
    for (BlockArgument arg : oldEntry.getArguments()) {
      if (auto attr = func.getArgAttrOfType<IntegerAttr>(
              arg.getArgNumber(), "bits.value_id")) {
        valueToValueId[arg] = attr.getInt();
      }
    }

    int64_t outputValueId = -1;
    if (assembleOp) {
      auto it = valueToValueId.find(assembleOp.getInput());
      if (it != valueToValueId.end()) {
        outputValueId = it->second;
      }
    }

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

    DenseMap<int64_t, TypedValue<SliceType>> valueIdToSlice;
    for (Operation *oldOp : toKeep) {
      auto transpose = cast<TransposeOp>(oldOp);
      auto it = valueToValueId.find(transpose.getOutput());
      if (it != valueToValueId.end()) {
        auto newSlice =
            cast<TypedValue<SliceType>>(mapping.lookup(transpose.getOutput()));
        valueIdToSlice[it->second] = newSlice;
      }
    }

    oldEntry.dropAllReferences();
    oldEntry.erase();

    DenseMap<int64_t, Value> i64Vals;
    auto getOrCreateI64Val = [&opBuilder, &loc, &i64Type, &i64Vals](int64_t num)
        -> Value {
      if (!i64Vals.contains(num)) {
        Value val = opBuilder.create<arith::ConstantOp>(
            loc, i64Type, opBuilder.getI64IntegerAttr(num));
        i64Vals[num] = val;
      }
      return i64Vals.lookup(num);
    };

    auto lowerNetwork = [&](MIG localMig,
                            ArrayRef<TypedValue<SliceType>> localInputs,
                            const DenseMap<int, int> &localCarryMap,
                            bool localIsMulF,
                            bool localIsMax,
                            bool localIsMin,
                            int64_t bankId)
        -> FailureOr<TypedValue<SliceType>> {
      const auto program_str = getProgram(localMig);
      ProgramParser parser(program_str);
      if (failed(parser.parse())) {
        return failure();
      }
      std::vector<Instruction> program = parser.getProgram();

      const int inputNum = localInputs.size();
      const bool localIsRed = inputNum == 1;
      const int carryNum = localCarryMap.size();

      AddressAllocator allocator(bankId);

      llvm::StringMap<RowAddress> allocated;
      llvm::StringMap<TypedValue<RowType>> firstRows;
      llvm::StringMap<TypedValue<RowType>> rowMap;
      llvm::StringMap<TypedValue<RowType>> carryRows;

      RowAddress outputFirstRowAddress;
      TypedValue<RowType> outputFirstRow;
      SmallVector<TypedValue<SliceType>> outputSlices;
      TypedValue<RowType> maskRow;

      const auto bRowType = RowType::get(ctx, 0);
      const auto cRowType = RowType::get(ctx, 1);
      const auto dRowType = RowType::get(ctx, 2);

      auto getOrCreateDRow = [&loc, &opBuilder, &dRowType, &rowMap,
                              &getOrCreateI64Val](RowAddress addr)
          -> TypedValue<RowType> {
        auto addrStr = addr.str();
        if (!rowMap.contains(addrStr)) {
          Value rowVal = opBuilder.create<GetRowOp>(
              loc, dRowType, getOrCreateI64Val(addr.channel),
              getOrCreateI64Val(addr.rank), getOrCreateI64Val(addr.bank),
              getOrCreateI64Val(addr.subarray), getOrCreateI64Val(addr.row));
          rowMap[addrStr] = cast<TypedValue<RowType>>(rowVal);
        }
        return rowMap.lookup(addrStr);
      };

      const Value zero = getOrCreateI64Val(0);
      const Value bankVal = getOrCreateI64Val(bankId);

      // =========================================================================
      // ================= Bitwise & Control Rows Generation =====================
      // TODO: This is a workaround.
      //       Currently we use B and C group rows from subarry (0, 0, 0, 0) only.
      //       We need an algorithm to find the optimality with lowest latency of
      //       inter-subarray row clone.
      // =========================================================================

      SmallVector<TypedValue<RowType>, 16> bRows;
      for (int i = 0; i < 16; ++i) {
        Value bRow = opBuilder.create<GetRowOp>(
            loc, bRowType, zero, zero, bankVal, zero, getOrCreateI64Val(i));
        bRows.push_back(cast<TypedValue<RowType>>(bRow));
      }

      const TypedValue<RowType> c0 = opBuilder.create<GetRowOp>(
          loc, cRowType, zero, zero, bankVal, zero, zero);
      const TypedValue<RowType> c1 = opBuilder.create<GetRowOp>(
          loc, cRowType, zero, zero, bankVal, zero, getOrCreateI64Val(1));

      // =========================================================================
      // ====================== Data Rows Allocation =============================
      // =========================================================================

      const auto maxColNum = allocator.getMaxColumnNum();
      int lastSliceVecLen = vecLen % maxColNum;
      if (lastSliceVecLen == 0)
        lastSliceVecLen = maxColNum;
      int round = (vecLen - 1) / maxColNum + 1;
      const Value maxColVal = getOrCreateI64Val(maxColNum);
      auto standardSliceType = localIsRed
          ? SliceType::get(ctx, 1, maxColNum)
          : SliceType::get(ctx, bitWidth, maxColNum);

      SmallVector<SmallVector<TypedValue<SliceType>>> slicesToStore;
      for (const auto &slice : localInputs) {
        SmallVector<TypedValue<SliceType>> v;
        v.push_back(slice);
        for (int roundIdx = 0; roundIdx < round - 1; ++roundIdx) {
          auto toSplit = v[roundIdx];
          auto secondColNum = toSplit.getType().getVectorLength() - maxColNum;
          if (roundIdx + 2 == round)
            assert(secondColNum <= maxColNum);
          auto secondType = SliceType::get(ctx, bitWidth, secondColNum);
          auto splitOp = opBuilder.create<SplitSliceVerticallyOp>(
              loc, standardSliceType, secondType, toSplit, maxColVal);
          v[roundIdx] = splitOp.getFirst();
          v.push_back(splitOp.getSecond());
        }
        slicesToStore.push_back(v);
      }

      for (int roundIdx = 0; roundIdx < round; ++roundIdx) {
        allocator.reset(bankId);
        StringSet<> stored;

      // std::vector<Instruction> program_gtLt;
      // if (isMax || isMin) {
      //   program_gtLt = isMax ? program_gt : program_lt;
      //   for (const auto &inst : program_gtLt) {
      //     if (inst.type == Instruction::Type::AAP) {
      //       auto operand = *inst.operand1;
      //       if (operand.str_repr == "O0") {
      //         auto maskRowAddr = allocator.allocate(1);
      //         maskRow = getOrCreateDRow(maskRowAddr);
      //         break;
      //       }
      //     }
      //   }
      // }

        for (auto &inst : program) {
        if (inst.type == Instruction::Type::AAP) {
          auto operand0 = inst.operand0;
          if (operand0.type == AddressType::Data) {
            const auto inputIdx = std::get<int>(operand0.data);
            if (inputIdx < inputNum) {
              if (!allocated.contains(operand0.str_repr)) {
                assert(roundIdx == 0);
                auto addr = allocator.allocate(bitWidth);
                allocated[operand0.str_repr] = addr;
                auto firstRow = getOrCreateDRow(addr);
                firstRows[operand0.str_repr] = firstRow;
              }
              if (!stored.contains(operand0.str_repr)) {
                auto data = std::get<int>(operand0.data);
                auto toStore = slicesToStore[data][roundIdx];
                assert(firstRows.contains(operand0.str_repr));
                opBuilder.create<StoreOp>(
                    loc, toStore, firstRows.lookup(operand0.str_repr));
                stored.insert(operand0.str_repr);
              }
            } else {
              if (!allocated.contains(operand0.str_repr)) {
                assert(roundIdx == 0);
                if (localIsMax || localIsMin || localIsRed) {
                  // do nothing
                } else {
                  auto addr = allocator.allocate(1);
                  allocated[operand0.str_repr] = addr;
                  auto cinRow = getOrCreateDRow(addr);
                  assert(!carryRows.contains(operand0.str_repr));
                  carryRows.try_emplace(operand0.str_repr, cinRow);
                }
              }
            }
          }

          auto operand1 = *inst.operand1;
          if (operand1.type == AddressType::Data) {
            const auto index = std::get<int>(operand1.data);
            assert(index >= inputNum);
            if (index - inputNum < carryNum * 2 || localIsRed) {
              if (!allocated.contains(operand1.str_repr)) {
                assert(roundIdx == 0);
                if (localIsRed) {
                  auto addr = allocator.allocate(1);
                  allocated[operand1.str_repr] = addr;
                  auto outRow = getOrCreateDRow(addr);
                  outputFirstRow = outRow;
                } else {
                  assert(!carryRows.contains(operand1.str_repr));
                  std::string cinName = "D" + std::to_string(index - carryNum);
                  assert(carryRows.contains(cinName) && allocated.contains(cinName));
                  allocated.try_emplace(operand1.str_repr, allocated.lookup(cinName));
                  carryRows.try_emplace(operand1.str_repr, carryRows.lookup(cinName));
                }
              }
              continue;
            }
            if (!allocated.contains(operand1.str_repr) && index == inputNum + carryNum * 2) {
              assert(roundIdx == 0);
              auto addr = allocator.allocate(bitWidth);
              allocated[operand1.str_repr] = addr;
              auto firstRow = getOrCreateDRow(addr);
              firstRows[operand1.str_repr] = firstRow;
              if (index == inputNum + carryNum * 2) {
                outputFirstRowAddress = addr;
                outputFirstRow = firstRow;
              }
            }
          } else if (operand1.type == AddressType::Bitwise) {}
        } else {
          assert(inst.type == Instruction::Type::AP);
        }
        }

      // llvm::StringMap<RowAddress> biasSFirstRowMap;
      // if (isMulF) {
      //   // For exponent - bias
      //   biasFirstRowAddress = allocator.allocate(exponentBitWidth);
      //   auto biasFirstRow = getOrCreateDRow(biasFirstRowAddress);
      //   opBuilder.create<StoreOp>(loc, exponentBias, biasFirstRow);
      //   auto cinAddr = allocator.allocate(1);
      //   biasCinRow = getOrCreateDRow(cinAddr);
      //   auto coutAddr = allocator.allocate(1);
      //   biasCoutRow = getOrCreateDRow(coutAddr);
      //   for (const auto &inst : program_minus_bias) {
      //     if (inst.type == Instruction::Type::AAP) {
      //       auto operand = *(inst.operand1);
      //       if (operand.type == AddressType::Spill) {
      //         if (!biasSFirstRowMap.contains(operand.str_repr)) {
      //           auto sAddr = allocator.allocate(1);
      //           biasSFirstRowMap[operand.str_repr] = sAddr;
      //         }
      //       }
      //     }
      //   }
      // }

        if (this->unroll) {

        // Sign-bit Computation
        // if (isMulF) {
        //   for (const auto &inst : program_mulf_sign) {
        //     if (inst.type == Instruction::Type::AP) {
        //       assert(inst.operand0.type == AddressType::Bitwise);
        //       auto index = std::get<int>(inst.operand0.data);
        //       opBuilder.create<APOp>(loc, bRows[index]);
        //     } else {
        //       assert(inst.type == Instruction::Type::AAP);

        //       TypedValue<RowType> addr0, addr1;

        //       auto operand0 = inst.operand0;
        //       auto operand1 = *inst.operand1;

        //       if (operand0.type == AddressType::Bitwise) {
        //         auto index = std::get<int>(operand0.data);
        //         addr0 = bRows[index];
        //       } else if (operand0.type == AddressType::Const) {
        //         addr0 = std::get<bool>(operand0.data) ? c1 : c0;
        //       } else {
        //         assert(operand0.type == AddressType::In ||
        //             operand0.type == AddressType::Spill);
        //         addr0 = firstRows.lookup(operand0.str_repr);
        //       }

        //       if (operand1.type == AddressType::Bitwise) {
        //         auto index = std::get<int>(operand1.data);
        //         addr1 = bRows[index];
        //       } else if (operand1.type == AddressType::Spill) {
        //         addr1 = firstRows.lookup(operand1.str_repr);
        //       } else {
        //         assert(operand1.type == AddressType::Out);
        //         addr1 = outputFirstRow;
        //       }

        //       opBuilder.create<AAPOp>(loc, addr0, addr1);

        //     }
        //   }
        // }

        // if (isMax || isMin) {
        //   program_gtLt = isMax ? program_gt : program_lt;
        //   int gtLtIdx = bitWidth;
        //   opBuilder.create<AAPOp>(loc, c0, bRows[2]);
        //   auto firstAddrIn0 = allocated["I0"];
        //   auto firstAddrIn1 = allocated["I1"];
        //   while (gtLtIdx > 0) {
        //     gtLtIdx--;
          //   for (const auto &inst : program_gtLt) {
          //     if (inst.type == Instruction::Type::AAP) {
          //       TypedValue<RowType> addr0, addr1;
          //       auto operand0 = inst.operand0;
          //       if (operand0.type == AddressType::Bitwise) {
          //         addr0 = bRows[std::get<int>(operand0.data)];
          //       } else if (operand0.type == AddressType::Const) {
          //         assert(std::get<bool>(operand0.data) == false);
          //         addr0 = c0;
          //       } else {
          //         assert(operand0.type == AddressType::In);
          //         assert(allocated.contains(operand0.str_repr));
          //         auto firstRow = allocated[operand0.str_repr];
          //         if (gtLtIdx == 0) {
          //           addr0 = getOrCreateDRow(firstRow);
          //         } else {
          //           addr0 = getOrCreateDRow(
          //               allocator.getRowFromOffset(firstRow, gtLtIdx));
          //         }
          //       }

          //       auto operand1 = *inst.operand1;
          //       if (operand1.type == AddressType::Bitwise) {
          //         addr1 = bRows[std::get<int>(operand1.data)];
          //       } else {
          //         assert(operand1.str_repr == "O0");
          //         addr1 = maskRow;
          //         std::cout << "Write to Mask\n";
          //       }

          //       opBuilder.create<AAPOp>(loc, addr0, addr1);
          //     } else {
          //       TypedValue<RowType> addr = bRows[std::get<int>(inst.operand0.data)];
          //       opBuilder.create<APOp>(loc, addr);
          //     }
          //   }
          // }
        //     TypedValue<RowType> in0, in1;
        //     if (gtLtIdx == 0) {
        //       in0 = getOrCreateDRow(firstAddrIn0);
        //       in1 = getOrCreateDRow(firstAddrIn1);
        //     } else {
        //       in0 = getOrCreateDRow(allocator.getRowFromOffset(firstAddrIn0, gtLtIdx));
        //       in1 = getOrCreateDRow(allocator.getRowFromOffset(firstAddrIn1, gtLtIdx));
        //     }
        //     if (isMax) {
        //       opBuilder.create<AAPOp>(loc, in0, bRows[1]);
        //       opBuilder.create<AAPOp>(loc, in1, bRows[5]);
        //     } else {
        //       assert(isMin);
        //       opBuilder.create<AAPOp>(loc, in0, bRows[5]);
        //       opBuilder.create<AAPOp>(loc, in1, bRows[1]);
        //     }
        //     opBuilder.create<APOp>(loc, bRows[14]);
        //   }
        //   opBuilder.create<AAPOp>(loc, bRows[2], maskRow);
        // }

        int iterCount = localIsMulF ? bitWidth - 1 : bitWidth;

        int64_t iterIndex = 0;
        int64_t addrOffset = bitWidth;
        while (iterIndex < iterCount) {
          addrOffset--;
          if (iterIndex == 2 && localIsRed)
            addrOffset--;
          llvm::StringSet<> refreshedCin;
          TypedValue<RowType> i1RowForReduction;
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
                assert(operand0.type == AddressType::Data);
                if ((localIsMax || localIsMin) && operand0.str_repr == "I2") {
                  addr0 = maskRow;
                } else if (localIsRed && operand0.str_repr == "I1") {
                  if (iterIndex == 0) {
                    if (!i1RowForReduction) {
                      auto firstRow = allocated["I0"];
                      addr0 = getOrCreateDRow(
                          allocator.getRowFromOffset(firstRow, addrOffset - 1));
                    } else {
                      addr0 = i1RowForReduction;
                    }
                  } else {
                    continue;
                  }
                } else {
                  auto firstRow = allocated[operand0.str_repr];
                  if (carryRows.contains(operand0.str_repr)) {
                    if (iterIndex == 0) {
                      addr0 = c0;
                    } else {
                      addr0 = carryRows.lookup(operand0.str_repr);
                      // if (!refreshedCin.contains(operand0.str_repr)) {
                      //   const auto cinIdx = std::get<int>(operand0.data);
                      //   const auto coutIdx = carryMap.lookup(cinIdx);
                      //   auto carry = carryRows.lookup(
                      //       std::format("O{}", coutIdx));
                      //   opBuilder.create<AAPOp>(loc, carry, addr0);
                      //   refreshedCin.insert(operand0.str_repr);
                      // }
                    }
                  } else {
                    if (addrOffset == 0) {
                      assert(firstRows.contains(operand0.str_repr));
                      addr0 = firstRows.lookup(operand0.str_repr);
                    } else {
                      addr0 = getOrCreateDRow(
                          allocator.getRowFromOffset(firstRow, addrOffset));
                    }
                  }
                }
              }

              if (operand1.type == AddressType::Bitwise) {
                auto index = std::get<int>(operand1.data);
                addr1 = bRows[index];
              } else {
                assert(operand1.type == AddressType::Data);
                if (localIsRed) {
                  if (iterIndex == iterCount - 1) {
                    addr1 = outputFirstRow;
                  } else {
                    if (program.size() > 4) {
                      addr1 = bRows[7];
                      // continue;
                    } else {
                      opBuilder.create<APOp>(loc, addr0);
                      continue;
                    }
                  }
                } else if (carryRows.contains(operand1.str_repr)) {
                  const auto outputIdx = std::get<int>(operand1.data);
                  addr1 = carryRows.lookup(operand1.str_repr);
                } else {
                  auto firstRow = allocated[operand1.str_repr];
                  if (addrOffset == 0) {
                    assert(firstRows.contains(operand1.str_repr));
                    addr1 = firstRows.lookup(operand1.str_repr);
                  } else {
                    addr1 = getOrCreateDRow(
                        allocator.getRowFromOffset(firstRow, addrOffset));
                  }
                }
              }

              opBuilder.create<AAPOp>(loc, addr0, addr1);

            }
          }

          iterIndex++;
          if (iterIndex == 1 && localIsRed)
            iterIndex++;

        }

        // exponent - bias
        // for (int biasOffset = exponentBitWidth; biasOffset > 0;
        //     --biasOffset) {
        //   bool cinRefreshed = false;
        //   for (const auto &inst : program_minus_bias) {
        //     if (inst.type == Instruction::Type::AP) {
        //       assert(inst.operand0.type == AddressType::Bitwise);
        //       auto index = std::get<int>(inst.operand0.data);
        //       opBuilder.create<APOp>(loc, bRows[index]);
        //     } else {
        //       assert(inst.type == Instruction::Type::AAP);

        //       TypedValue<RowType> addr0, addr1;

        //       auto operand0 = inst.operand0;
        //       auto operand1 = *inst.operand1;

        //       if (operand0.type == AddressType::Bitwise) {
        //         auto index = std::get<int>(operand0.data);
        //         addr0 = bRows[index];
        //       } else if (operand0.type == AddressType::Const) {
        //         addr0 = std::get<bool>(operand0.data) ? c1 : c0;
        //       } else if (operand0.type == AddressType::Spill) {
        //         assert(biasSFirstRowMap.contains(operand0.str_repr));
        //         addr0 = getOrCreateDRow(biasSFirstRowMap[operand0.str_repr]);
        //       } else {
        //         assert(operand0.type == AddressType::In);
        //         if (operand0.str_repr == "I0") {
        //           addr0 = getOrCreateDRow(allocator.getRowFromOffset(
        //               outputFirstRowAddress, biasOffset));
        //         } else if (operand0.str_repr == "I1") {
        //           addr0 = getOrCreateDRow(allocator.getRowFromOffset(
        //               biasFirstRowAddress, biasOffset - 1));
        //         } else {
        //           assert(operand0.str_repr == "I2");
        //           if (biasOffset == exponentBitWidth) {
        //             addr0 = c0;
        //           } else {
        //             if (!cinRefreshed) {
        //               opBuilder.create<AAPOp>(loc, biasCoutRow, biasCinRow);
        //               cinRefreshed = true;
        //             }
        //             addr0 = biasCinRow;
        //           }
        //         }
        //       }

        //       if (operand1.type == AddressType::Bitwise) {
        //         auto index = std::get<int>(operand1.data);
        //         addr1 = bRows[index];
        //       } else if (operand1.type == AddressType::Spill) {
        //         assert(biasSFirstRowMap.contains(operand0.str_repr));
        //         addr1 = getOrCreateDRow(biasSFirstRowMap[operand0.str_repr]);
        //       } else {
        //         assert(operand1.type == AddressType::Out);
        //         if (operand1.str_repr == "O0") {
        //           addr1 = biasCoutRow;
        //         } else {
        //           assert(operand1.str_repr == "O1");
        //           addr1 = getOrCreateDRow(allocator.getRowFromOffset(
        //               outputFirstRowAddress, biasOffset));
        //         }
        //       }

        //       opBuilder.create<AAPOp>(loc, addr0, addr1);

        //     }
        //   }
        // }

        } else {

        auto loop = opBuilder.create<affine::AffineForOp>(loc, 0, bitWidth, 1);
        opBuilder.setInsertionPointToStart(loop.getBody());
        Value iterIndex = opBuilder.create<arith::IndexCastOp>(
            loc, i64Type, loop.getInductionVar());
        const auto maxOffset = getOrCreateI64Val(bitWidth - 1);
        const Value offset = opBuilder.create<arith::SubIOp>(
            loc, i64Type, maxOffset, iterIndex);

        for (const auto &inst : program) {
          const auto &operand0 = inst.operand0;
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
            assert(operand0.type == AddressType::Data);
            auto row = allocated.lookup(operand0.str_repr);
            Value rowID = opBuilder.create<arith::AddIOp>(
                loc, i64Type, getOrCreateI64Val(row.row), offset);
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
            assert(operand1.type == AddressType::Data);
            auto row = allocated.lookup(operand1.str_repr);
            Value rowID = opBuilder.create<arith::AddIOp>(
                loc, i64Type, getOrCreateI64Val(row.row), offset);
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

        SliceType sType = standardSliceType;
        if (roundIdx + 1 == round) {
          sType = localIsRed ? SliceType::get(ctx, 1, lastSliceVecLen)
              : SliceType::get(ctx, bitWidth, lastSliceVecLen);
        }
        TypedValue<SliceType> slice = opBuilder.create<LoadOp>(
            loc, sType, outputFirstRow, getOrCreateI64Val(bitWidth));
        outputSlices.push_back(slice);
      }

      auto resultSlice = outputSlices[0];
      if (round > 1) {
        for (int i = 1; i < round; ++i) {
          auto second = outputSlices[i];
          auto vecLen = resultSlice.getType().getVectorLength() +
              second.getType().getVectorLength();
          auto sType = localIsRed ? SliceType::get(ctx, 1, vecLen)
              : SliceType::get(ctx, bitWidth, vecLen);
          resultSlice = opBuilder.create<MergeSliceVerticallyOp>(
              loc, sType, resultSlice, second);
        }
      }

      return resultSlice;
    };

    FailureOr<TypedValue<SliceType>> maybeResult;
    if (builder.hasSubgraphs()) {
      DenseMap<int64_t, const NetworkBuilder::SubgraphNetwork *> bankToSubgraph;
      for (const auto &subgraph : builder.getSubgraphNetworks()) {
        bankToSubgraph[subgraph.bankId] = &subgraph;
      }

      DenseMap<int64_t, SmallVector<int64_t>> adjacency;
      DenseMap<int64_t, int> indegree;
      for (const auto &entry : bankToSubgraph) {
        indegree[entry.first] = 0;
      }

      for (const auto &dep : builder.getSubgraphDependencies()) {
        int64_t producer = static_cast<int64_t>(dep.producerBankId);
        int64_t consumer = static_cast<int64_t>(dep.consumerBankId);
        adjacency[producer].push_back(consumer);
        indegree[consumer] += 1;
      }

      SmallVector<int64_t> ready;
      for (const auto &entry : indegree) {
        if (entry.second == 0) {
          ready.push_back(entry.first);
        }
      }
      llvm::sort(ready);

      SmallVector<int64_t> orderedBanks;
      while (!ready.empty()) {
        int64_t bankId = ready.pop_back_val();
        orderedBanks.push_back(bankId);
        for (int64_t consumer : adjacency[bankId]) {
          auto &count = indegree[consumer];
          count -= 1;
          if (count == 0) {
            ready.push_back(consumer);
            llvm::sort(ready);
          }
        }
      }

      if (orderedBanks.size() != bankToSubgraph.size()) {
        orderedBanks.clear();
        for (const auto &subgraph : builder.getSubgraphNetworks()) {
          orderedBanks.push_back(subgraph.bankId);
        }
      }

      for (int64_t bankId : orderedBanks) {
        auto it = bankToSubgraph.find(bankId);
        if (it == bankToSubgraph.end()) {
          continue;
        }
        const auto &subgraph = *it->second;
        SmallVector<TypedValue<SliceType>> subgraphInputs;
        for (int64_t valueId : subgraph.inputValueIds) {
          if (!valueIdToSlice.contains(valueId)) {
            signalPassFailure();
            return;
          }
          subgraphInputs.push_back(valueIdToSlice.lookup(valueId));
        }

        auto resultSlice = lowerNetwork(
            subgraph.mig, subgraphInputs, subgraph.carryMap, subgraph.isMulF,
            subgraph.isMax, subgraph.isMin, subgraph.bankId);
        if (failed(resultSlice)) {
          signalPassFailure();
          return;
        }

        for (int64_t valueId : subgraph.outputValueIds) {
          valueIdToSlice[valueId] = *resultSlice;
        }
      }
    } else {
      SmallVector<TypedValue<SliceType>> mappedInputs;
      mappedInputs.reserve(ntkInputs.size());
      for (const auto &oldSlice : ntkInputs) {
        assert(mapping.contains(oldSlice));
        mappedInputs.push_back(
            cast<TypedValue<SliceType>>(mapping.lookup(oldSlice)));
      }
      maybeResult = lowerNetwork(mig, mappedInputs, carryMap, isMulF, isMax,
                                 isMin, /*bankId=*/0);
      if (failed(maybeResult)) {
        signalPassFailure();
        return;
      }
    }

    TypedValue<SliceType> resultSlice;
    if (builder.hasSubgraphs()) {
      if (outputValueId < 0 || !valueIdToSlice.contains(outputValueId)) {
        signalPassFailure();
        return;
      }
      resultSlice = valueIdToSlice.lookup(outputValueId);
    } else {
      resultSlice = *maybeResult;
    }

    Value resultTensor =
        opBuilder.create<AssembleOp>(loc, tensorType, resultSlice);
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
