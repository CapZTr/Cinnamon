//===- BitsRowsToPuD.cpp - row-level Bits to PuD row ops ------------------===//
//
// Turns row-level Bits ops into the row-op sequences the substrate executes, by
// instantiating the templates RowOpLibrary compiled for it.
//
// The input is what `bits-pattern-apply` leaves behind: a loop over bit rows
// whose body holds one row op, nested in a tile loop that `bits-opt-mapping`
// annotated with the banks it was mapped onto. The output is a function that
// does nothing but issue row ops -- the bit-serial schedule, and nothing of the
// tensor dataflow that described it.
//
// Discarding that dataflow is deliberate rather than a shortcut. A row op has no
// value semantics to preserve: it names DRAM rows and leaves its result in them,
// so once operands are bound to slots the SSA values that stood for rows have
// nothing left to say. What has to be preserved is the ORDER, and specifically
// that each command reaches every mapped bank before the next command starts --
// that is what lets banks overlap their activations, and it is the order the
// trace player replays, so both paths describe the same execution.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

using namespace mlir;
using namespace mlir::bits;

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/BitsPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Slot layout
//===----------------------------------------------------------------------===//

/// Assigns row slots, mirroring RowOpTracePlayer's layout so both paths address
/// the same rows: slots 0-17 are the control rows init_ambit() sets up, and the
/// data pools follow from slot 18.
///
/// A pool is as wide as the operand, because a bit-serial op reads bit j of its
/// operands from row `base + j`. The budget is what the guest can keep
/// physically contiguous, which is one subarray.
struct SlotLayout {
  static constexpr int kMaxSlots = 512;

  int lhsBase = SLOT_DATA_BASE;
  int rhsBase = 0;
  int outBase = 0;
  int tempBase = 0;
  int total = 0;

  static FailureOr<SlotLayout> build(int width, int numTemps,
                                     Operation *reportOn) {
    SlotLayout layout;
    layout.lhsBase = SLOT_DATA_BASE;
    layout.rhsBase = layout.lhsBase + width;
    layout.outBase = layout.rhsBase + width;
    layout.tempBase = layout.outBase + width;
    layout.total = layout.tempBase + numTemps;
    if (layout.total > kMaxSlots) {
      reportOn->emitError()
          << "row slot budget exhausted: " << layout.total
          << " slots needed but a subarray holds " << kMaxSlots
          << "; the guest cannot keep more than that physically contiguous";
      return failure();
    }
    return layout;
  }
};

//===----------------------------------------------------------------------===//
// The schedule to emit
//===----------------------------------------------------------------------===//

/// What the mapped row-level IR says the kernel does, reduced to the parameters
/// code generation needs.
struct RowSchedule {
  /// Row-op primitive the kernel's bit loop runs.
  RowOpKind kind = RowOpKind::FullAdd;
  /// Bits per operand, i.e. the bit loop's trip count.
  int64_t width = 0;
  /// First bank and how many banks run in parallel.
  int64_t bankBase = 0;
  int64_t bankCount = 1;
  /// Tiles each bank processes in sequence.
  int64_t itersPerBank = 1;
  /// Row op that carries state across bit iterations in a dual-contact cell,
  /// and therefore needs seeding once per tile. Zero-initialised for an add
  /// (the carry starts at 0); the slot is the cell the template pinned it to.
  bool seedsCarry = false;
  int carrySlot = SLOT_DCC1;
  int carrySeedSlot = SLOT_C_0;
};

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

class RowOpEmitter {
public:
  RowOpEmitter(OpBuilder &builder, const SlotLayout &layout)
      : builder(builder), layout(layout) {}

  /// Issues one command across `[bankBase, bankBase + bankCount)`. Banks are the
  /// inner loop so one command reaches every bank before the next starts.
  void emitAcrossBanks(Location loc, bool isAp, Value dstSlot, Value srcSlot,
                       int64_t bankBase, int64_t bankCount) {
    Value lb = builder.create<arith::ConstantIndexOp>(loc, bankBase);
    Value ub = builder.create<arith::ConstantIndexOp>(loc, bankBase + bankCount);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    builder.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
          Value bank =
              b.create<arith::IndexCastOp>(bodyLoc, b.getI64Type(), iv);
          Value dst = b.create<pud::RowAddrOp>(
              bodyLoc, rowTypeFor(b, dstSlot), dstSlot, bank);
          if (isAp) {
            b.create<pud::APOp>(bodyLoc, dst);
          } else {
            Value src = b.create<pud::RowAddrOp>(
                bodyLoc, rowTypeFor(b, srcSlot), srcSlot, bank);
            b.create<pud::AAPOp>(bodyLoc, src, dst);
          }
          b.create<scf::YieldOp>(bodyLoc);
        });
  }

  /// Instantiates `tmpl` for one bit position. `bitIv` indexes the bit row, so
  /// the template's i-th data input reads from `pool + bitIv`.
  LogicalResult emitTemplate(Location loc, const RowOpTemplate &tmpl,
                             Value bitIv, const RowSchedule &sched) {
    auto slotForBit = [&](int base) -> Value {
      Value baseVal = builder.create<arith::ConstantIntOp>(loc, base, 64);
      return builder.create<arith::AddIOp>(loc, baseVal, bitIv);
    };
    SmallVector<Value, 3> inputs{slotForBit(layout.lhsBase),
                                 slotForBit(layout.rhsBase)};
    SmallVector<Value, 2> outputs{slotForBit(layout.outBase)};

    for (const RowOpInstr &instr : tmpl.instrs) {
      Value dst = resolve(loc, instr.dst, inputs, outputs);
      Value src = instr.isAp ? Value() : resolve(loc, instr.src, inputs, outputs);
      if (!dst || (!instr.isAp && !src))
        return failure();
      emitAcrossBanks(loc, instr.isAp, dst, src, sched.bankBase,
                      sched.bankCount);
    }
    return success();
  }

private:
  Value resolve(Location loc, const RowOperand &operand,
                ArrayRef<Value> inputs, ArrayRef<Value> outputs) {
    switch (operand.kind) {
    case RowOperand::Kind::Control:
      return constant(loc, operand.index);
    case RowOperand::Kind::Input:
      return operand.index < (int)inputs.size() ? inputs[operand.index]
                                                : Value();
    case RowOperand::Kind::Output:
      return operand.index < (int)outputs.size() ? outputs[operand.index]
                                                 : Value();
    case RowOperand::Kind::Temp:
      return constant(loc, layout.tempBase + operand.index);
    }
    return Value();
  }

  /// RowType's group is what the AAP/AP verifiers check: 0 for the bitwise
  /// control rows and their activation groups, 1 for the two constant rows,
  /// 2 for data. A slot that is not a compile-time constant indexes a data pool.
  pud::RowType rowTypeFor(OpBuilder &b, Value slot) {
    int64_t group = 2;
    if (auto constOp = slot.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t idx = constOp.value();
      if (idx == SLOT_C_0 || idx == SLOT_C_1)
        group = 1;
      else if (idx < SLOT_DATA_BASE)
        group = 0;
    }
    return pud::RowType::get(b.getContext(), group);
  }

  Value constant(Location loc, int64_t value) {
    return builder.create<arith::ConstantIntOp>(loc, value, 64);
  }

  OpBuilder &builder;
  const SlotLayout &layout;
};

//===----------------------------------------------------------------------===//
// Reading the schedule off the mapped IR
//===----------------------------------------------------------------------===//

int64_t getStaticTripCount(scf::ForOp loop) {
  auto lb = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb || !ub || !step || step.value() == 0)
    return 0;
  return (ub.value() - lb.value() + step.value() - 1) / step.value();
}

FailureOr<RowSchedule> readSchedule(func::FuncOp func) {
  SmallVector<Operation *> rowOps;
  func.walk([&](Operation *op) {
    if (isa<RowAddOp>(op))
      rowOps.push_back(op);
  });
  if (rowOps.empty()) {
    func.emitError() << "no row-level Bits ops to lower; run "
                        "bits-pattern-apply first";
    return failure();
  }
  if (rowOps.size() != 1) {
    rowOps.front()->emitError()
        << "this pass currently lowers a single row op per function, found "
        << rowOps.size();
    return failure();
  }

  Operation *rowOp = rowOps.front();
  RowSchedule sched;
  sched.kind = RowOpKind::FullAdd;
  sched.seedsCarry = true;

  auto bitLoop = rowOp->getParentOfType<scf::ForOp>();
  if (!bitLoop) {
    rowOp->emitError() << "expected the row op to sit in a loop over bit rows";
    return failure();
  }
  sched.width = getStaticTripCount(bitLoop);
  if (sched.width <= 0) {
    rowOp->emitError() << "expected a statically bounded loop over bit rows";
    return failure();
  }

  // The bank mapping sits on an enclosing loop; take the innermost that has it.
  bool found = false;
  for (Operation *parent = bitLoop.getOperation(); parent;
       parent = parent->getParentOfType<scf::ForOp>()) {
    auto pf = parent->getAttrOfType<IntegerAttr>("bits.parallel_factor");
    if (!pf)
      continue;
    sched.bankCount = pf.getInt();
    if (auto bid = parent->getAttrOfType<IntegerAttr>("bits.bank_id"))
      sched.bankBase = bid.getInt();
    if (auto ipb = parent->getAttrOfType<IntegerAttr>("bits.iters_per_bank"))
      sched.itersPerBank = ipb.getInt();
    found = true;
    break;
  }
  if (!found) {
    rowOp->emitError() << "no bank mapping on any enclosing loop; run "
                          "bits-opt-mapping first";
    return failure();
  }
  return sched;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertBitsRowsToPuD
    : public ConvertBitsRowsToPuDBase<ConvertBitsRowsToPuD> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    PudBackend target;
    if (backend == "simdram")
      target = PudBackend::Simdram;
    else if (backend == "prada")
      target = PudBackend::Prada;
    else if (backend == "fcdram")
      target = PudBackend::Fcdram;
    else {
      func.emitError("convert-bits-rows-to-pud: backend must be 'simdram', "
                     "'prada' or 'fcdram', got '")
          << backend << "'";
      return signalPassFailure();
    }

    auto sched = readSchedule(func);
    if (failed(sched))
      return signalPassFailure();

    // The full adder's carry is loop-carried through a dual-contact cell, so the
    // template that applies is the 8-command one, not the 9-command variant a
    // multiplier needs for carries that live in data rows.
    const RowOpTemplate *tmpl =
        RowOpLibrary::get().getTemplate(target, sched->kind);
    if (!tmpl) {
      func.emitError() << "no " << getRowOpName(sched->kind)
                       << " template for backend '" << getBackendName(target)
                       << "'; lime has no architecture definition for it";
      return signalPassFailure();
    }

    auto layout = SlotLayout::build(sched->width, /*numTemps=*/4, func);
    if (failed(layout))
      return signalPassFailure();

    // Replace the body wholesale: the tensor dataflow described which rows to
    // touch, and the slots now say it directly.
    Location loc = func.getLoc();
    Region &body = func.getBody();
    body.getBlocks().clear();
    Block *entry = new Block();
    body.push_back(entry);
    func.setType(FunctionType::get(func.getContext(), {}, {}));

    OpBuilder builder(entry, entry->begin());
    RowOpEmitter emitter(builder, *layout);

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value tiles = builder.create<arith::ConstantIndexOp>(loc, sched->itersPerBank);
    Value bits = builder.create<arith::ConstantIndexOp>(loc, sched->width);

    auto tileLoop = builder.create<scf::ForOp>(loc, zero, tiles, one);
    builder.setInsertionPointToStart(tileLoop.getBody());
    {
      RowOpEmitter tileEmitter(builder, *layout);
      if (sched->seedsCarry) {
        // One command per tile seeds the carry; the template keeps it resident
        // in the cell for the whole bit loop, which is what makes an n-bit add
        // 8n + 1 rather than 9n.
        Value carry =
            builder.create<arith::ConstantIntOp>(loc, sched->carrySlot, 64);
        Value seed =
            builder.create<arith::ConstantIntOp>(loc, sched->carrySeedSlot, 64);
        tileEmitter.emitAcrossBanks(loc, /*isAp=*/false, carry, seed,
                                    sched->bankBase, sched->bankCount);
      }

      auto bitLoop = builder.create<scf::ForOp>(loc, zero, bits, one);
      builder.setInsertionPointToStart(bitLoop.getBody());
      RowOpEmitter bitEmitter(builder, *layout);
      Value bitIv = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), bitLoop.getInductionVar());
      if (failed(bitEmitter.emitTemplate(loc, *tmpl, bitIv, *sched)))
        return signalPassFailure();
    }

    builder.setInsertionPointToEnd(entry);
    builder.create<func::ReturnOp>(loc);

    // The runtime has to allocate exactly the slots this layout assigned, so the
    // number travels with the IR rather than being restated by the build script.
    if (auto module = func->getParentOfType<ModuleOp>())
      module->setAttr("pud.num_slots",
                      builder.getI64IntegerAttr(layout->total));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::bits::createConvertBitsRowsToPuDPass() {
  return std::make_unique<ConvertBitsRowsToPuD>();
}
