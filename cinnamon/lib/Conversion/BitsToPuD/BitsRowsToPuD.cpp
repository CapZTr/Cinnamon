//===- BitsRowsToPuD.cpp - row-level Bits to PuD row ops ------------------===//
//
// Turns row-level Bits ops into the row-op sequences the substrate executes, by
// instantiating the templates RowOpLibrary compiled for it.
//
// The input is what `bits-pattern-apply` leaves behind: loops over bit rows
// whose bodies hold row ops, nested in a tile loop that `bits-opt-mapping`
// annotated with the banks it was mapped onto. The output is a function that
// does nothing but issue row ops -- the bit-serial schedule, and nothing of the
// tensor dataflow that described it.
//
// Between the two sits the question this pass exists to answer: which DRAM row
// does each row-level value live in. A `!bits.row` is an SSA value; a row op
// names rows and leaves its result in them. So every value gets bound to a
// place, and that binding is what the emitted addresses are:
//
//   * a slice is a run of consecutive rows, so `extract_row` is the run's first
//     row plus the index, and `insert_row` returns the run it was given;
//   * `broadcast_row` binds every index of its slice to one row -- it is a view,
//     and views cost nothing;
//   * a row a template produces is placed where its consumer needs it. When that
//     consumer is an `insert_row`, the template writes straight into the slice
//     rather than into scratch that would then have to be copied;
//   * a value carried across loop iterations has to keep still, so an iteration's
//     result is bound to the same row its predecessor was.
//
// A carry is the case that pays for all of this. When the whole chain -- the
// initial value, the loop-carried argument and the carry-out -- can be one
// place, it can be a dual-contact cell rather than a row, and the adder never
// spends a command reading it in or writing it back. That is 8 commands a bit
// instead of 9, and it is why `bits.carry` says which kind of adder an
// expansion built rather than leaving this pass to guess.
//
// Discarding the tensor dataflow is deliberate rather than a shortcut. A row op
// has no value semantics to preserve: once operands are bound to addresses, the
// SSA values that stood for rows have nothing left to say.
//
// What does have to be preserved is the order, and in particular that each
// command reaches every mapped bank before the next command starts. Banks are
// independent: their activations overlap, and issuing a command to all of them
// before moving on is what turns that independence into parallelism.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/BitsToPuD/BitsToPuD.h"
#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsTypes.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/Codegen/AddressAllocator.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include <memory>

using namespace mlir;
using namespace mlir::bits;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/BitsPasses.h.inc"

namespace {

/// Where a bits.row_add keeps its carry, as bits-pattern-apply recorded it.
constexpr llvm::StringLiteral kCarryAttrName = "bits.carry";
constexpr llvm::StringLiteral kCarryResident = "resident";

//===----------------------------------------------------------------------===//
// Where a value lives
//===----------------------------------------------------------------------===//

/// A row within one bank, with the bank left open -- which bank fills in per
/// iteration of the loop that issues the command.
struct RowLoc {
  bool valid = false;
  /// A row the substrate reserves, named rather than allocated.
  bool isControl = false;
  ControlRow control = ControlRow::T0;
  /// A data row: the first row of the run it is in, plus where in the run.
  int64_t base = 0;
  /// Index within the run, as a value of the source IR. Null means `offset`
  /// alone.
  Value index;
  int64_t offset = 0;

  static RowLoc controlRow(ControlRow row) {
    RowLoc loc;
    loc.valid = true;
    loc.isControl = true;
    loc.control = row;
    return loc;
  }
  static RowLoc dataRow(int64_t base, Value index = {}, int64_t offset = 0) {
    RowLoc loc;
    loc.valid = true;
    loc.base = base;
    loc.index = index;
    loc.offset = offset;
    return loc;
  }

  /// Whether two bindings name the same row without having to evaluate an
  /// index -- enough to tell a copy that would do nothing from one that would.
  bool sameAs(const RowLoc &o) const {
    if (!valid || !o.valid || isControl != o.isControl)
      return false;
    if (isControl)
      return control == o.control;
    return base == o.base && index == o.index && offset == o.offset;
  }
};

/// Where a slice's rows are: a run of consecutive rows, or one row presented at
/// every index.
struct SliceLoc {
  bool valid = false;
  bool broadcast = false;
  RowLoc row;
  int64_t base = 0;

  static SliceLoc run(int64_t base) {
    SliceLoc loc;
    loc.valid = true;
    loc.base = base;
    return loc;
  }
  static SliceLoc broadcastOf(const RowLoc &row) {
    SliceLoc loc;
    loc.valid = true;
    loc.broadcast = true;
    loc.row = row;
    return loc;
  }

  /// The row this slice presents at `index`.
  RowLoc at(Value index) const {
    return broadcast ? row : RowLoc::dataRow(base, index);
  }
};

/// A copy that has to happen before a loop, because the value a loop-carried
/// row starts from is not already in the row the loop keeps it in.
struct Seed {
  RowLoc dst;
  RowLoc src;
};

//===----------------------------------------------------------------------===//
// Binding values to rows
//===----------------------------------------------------------------------===//

/// Decides where every row-level value of a kernel lives.
///
/// One forward walk is enough: a definition precedes its uses, so when a
/// template's result needs a home the consumer that says where it should go is
/// already in the IR, and when a loop-carried value needs one the value it
/// starts from has already been bound.
class RowBinder {
public:
  RowBinder(pud::AddressAllocator &allocator, int64_t flatBank)
      : allocator(allocator), flatBank(flatBank) {}

  LogicalResult bind(Block &block);

  const RowLoc *rowFor(Value v) const {
    auto it = rows.find(v);
    return it == rows.end() ? nullptr : &it->second;
  }
  const SliceLoc *sliceFor(Value v) const {
    auto it = slices.find(v);
    return it == slices.end() ? nullptr : &it->second;
  }

  /// The run a slice belongs to, looking through inserts that have not been
  /// bound yet.
  ///
  /// An insert hands back the run it was given, so a chain of them is one run
  /// however far along it a value is asked about. Placement asks ahead of the
  /// walk -- where a row op's result should go is read off a consumer that has
  /// not been reached -- and that consumer's slice can be another such consumer.
  const SliceLoc *runFor(Value v) const {
    while (true) {
      if (const SliceLoc *loc = sliceFor(v))
        return loc;
      auto insert = v.getDefiningOp<InsertRowOp>();
      if (!insert)
        return nullptr;
      v = insert.getSlice();
    }
  }
  /// Copies that have to be issued before `loop` runs.
  ArrayRef<Seed> seedsFor(Operation *loop) const {
    auto it = seeds.find(loop);
    return it == seeds.end() ? ArrayRef<Seed>() : ArrayRef<Seed>(it->second);
  }

  /// Scratch rows the templates use internally, shared by every row op: a
  /// template's temporaries are dead the moment its last command retires.
  RowLoc tempRow(int index) {
    while ((int)temps.size() <= index)
      temps.push_back(allocator.allocate(flatBank, 1).row);
    return RowLoc::dataRow(temps[index]);
  }

private:
  RowLoc scratchRow() {
    return RowLoc::dataRow(allocator.allocate(flatBank, 1).row);
  }
  SliceLoc newRun(int64_t numRows) {
    return SliceLoc::run(allocator.allocate(flatBank, numRows).row);
  }

  /// Where a row a template produces has to end up, read off what consumes it.
  RowLoc homeFor(Value result);
  LogicalResult bindOp(Operation *op);
  LogicalResult bindLoop(scf::ForOp loop);

  pud::AddressAllocator &allocator;
  int64_t flatBank;
  DenseMap<Value, RowLoc> rows;
  DenseMap<Value, SliceLoc> slices;
  DenseMap<Operation *, SmallVector<Seed>> seeds;
  SmallVector<int64_t> temps;
};

/// Whether an op computes on rows, i.e. becomes a template instantiation.
bool isRowOp(Operation *op) {
  return isa<RowAddOp, RowAndOp, RowOrOp, RowXOrOp, RowXNorOp, MajOp,
             NotOp>(op);
}

bool hasResidentCarry(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>(kCarryAttrName);
  return attr && attr.getValue() == kCarryResident;
}

RowLoc RowBinder::homeFor(Value result) {
  // A result with one consumer that names a place belongs in that place; going
  // through scratch and copying would be a command per bit for nothing.
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (auto insert = dyn_cast<InsertRowOp>(user)) {
      if (use.getOperandNumber() != 1)
        continue;
      if (const SliceLoc *slice = runFor(insert.getSlice()))
        return slice->at(insert.getIndex());
    }
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      auto loop = dyn_cast<scf::ForOp>(yield->getParentOp());
      if (!loop)
        continue;
      // The loop keeps this value in the row its argument was bound to.
      Value arg = loop.getRegionIterArg(use.getOperandNumber());
      if (const RowLoc *home = rowFor(arg))
        return *home;
    }
  }
  return scratchRow();
}

LogicalResult RowBinder::bindLoop(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());

  for (auto [i, init] : llvm::enumerate(loop.getInits())) {
    Value arg = loop.getRegionIterArg(i);
    Value result = loop.getResult(i);

    if (isa<SliceType>(arg.getType())) {
      // A slice is updated in place: the loop hands back the run it was given.
      const SliceLoc *from = sliceFor(init);
      if (!from)
        return loop.emitError() << "loop carries a slice that is not bound to "
                                   "any rows";
      slices[arg] = *from;
      slices[result] = *from;
      continue;
    }
    if (!isa<BitRowType>(arg.getType()))
      continue;

    const RowLoc *from = rowFor(init);
    if (!from)
      return loop.emitError()
             << "loop carries a row that is not bound to any row";

    // A carry the adder can keep in its dual-contact cell never reaches a data
    // row at all: the chain is the cell, and the value it starts from is copied
    // in once, ahead of the loop.
    Value carried = yield.getOperand(i);
    auto *producer = carried.getDefiningOp();
    bool resident = producer && isa<RowAddOp>(producer) &&
                    hasResidentCarry(producer) &&
                    carried == cast<RowAddOp>(producer).getCout();

    RowLoc home = resident ? RowLoc::controlRow(ControlRow::DCC1)
                           : (from->isControl ? scratchRow() : *from);
    if (!home.sameAs(*from))
      seeds[loop.getOperation()].push_back({home, *from});

    rows[arg] = home;
    rows[result] = home;
  }
  return success();
}

LogicalResult RowBinder::bindOp(Operation *op) {
  // --- slices ---------------------------------------------------------------
  if (auto transpose = dyn_cast<TransposeOp>(op)) {
    auto slice = dyn_cast<SliceType>(transpose.getOutput().getType());
    if (!slice)
      return op->emitError() << "only a transpose to a slice can be placed";
    slices[transpose.getOutput()] = newRun(slice.getBitWidth());
    return success();
  }
  if (auto create = dyn_cast<CreateSliceOp>(op)) {
    auto slice = cast<SliceType>(create.getSlice().getType());
    slices[create.getSlice()] = newRun(slice.getBitWidth());
    return success();
  }
  if (auto ones = dyn_cast<CreateAllOneSliceOp>(op)) {
    slices[ones.getSlice()] =
        SliceLoc::broadcastOf(RowLoc::controlRow(ControlRow::C_1));
    return success();
  }
  if (auto bcast = dyn_cast<BroadcastRowOp>(op)) {
    const RowLoc *row = rowFor(bcast.getRow());
    if (!row)
      return op->emitError() << "broadcast of a row that is not bound";
    slices[bcast.getSlice()] = SliceLoc::broadcastOf(*row);
    return success();
  }
  if (auto sub = dyn_cast<ExtractSubSliceOp>(op)) {
    const SliceLoc *from = sliceFor(sub.getSlice());
    if (!from)
      return op->emitError() << "sub-slice of a slice that is not bound";
    if (from->broadcast) {
      slices[sub.getResultSlice()] = *from;
      return success();
    }
    APInt offset;
    if (!matchPattern(sub.getRowNum(), m_ConstantInt(&offset)))
      return op->emitError() << "a sub-slice has to start at a known row";
    slices[sub.getResultSlice()] =
        SliceLoc::run(from->base + offset.getSExtValue());
    return success();
  }
  if (auto insert = dyn_cast<InsertRowOp>(op)) {
    const SliceLoc *from = sliceFor(insert.getSlice());
    if (!from)
      return op->emitError() << "insert into a slice that is not bound";
    // The row was written where this insert says; the slice is unchanged.
    slices[insert.getResultSlice()] = *from;
    return success();
  }

  // --- rows -----------------------------------------------------------------
  if (auto zero = dyn_cast<CreateAllZeroRowOp>(op)) {
    rows[zero.getRow()] = RowLoc::controlRow(ControlRow::C_0);
    return success();
  }
  if (auto create = dyn_cast<CreateRowOp>(op)) {
    rows[create.getRow()] = scratchRow();
    return success();
  }
  if (auto extract = dyn_cast<ExtractRowOp>(op)) {
    const SliceLoc *from = sliceFor(extract.getSlice());
    if (!from)
      return op->emitError() << "extract from a slice that is not bound";
    rows[extract.getRow()] = from->at(extract.getRowIdx());
    return success();
  }

  // --- row ops --------------------------------------------------------------
  if (isRowOp(op)) {
    if (auto add = dyn_cast<RowAddOp>(op)) {
      rows[add.getSum()] = homeFor(add.getSum());
      rows[add.getCout()] = hasResidentCarry(op)
                                ? RowLoc::controlRow(ControlRow::DCC1)
                                : homeFor(add.getCout());
      return success();
    }
    rows[op->getResult(0)] = homeFor(op->getResult(0));
    return success();
  }

  return success();
}

LogicalResult RowBinder::bind(Block &block) {
  for (Operation &op : block) {
    if (failed(bindOp(&op)))
      return failure();
    if (auto loop = dyn_cast<scf::ForOp>(&op)) {
      if (failed(bindLoop(loop)))
        return failure();
      if (failed(bind(*loop.getBody())))
        return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Issues the commands of a template, one bank at a time.
class RowOpEmitter {
public:
  RowOpEmitter(const pud::MemoryGeometry &geometry, RowBinder &binder,
               int64_t bankBase, int64_t bankCount)
      : geometry(geometry), binder(binder), bankBase(bankBase),
        bankCount(bankCount) {}

  /// Issues one command to every mapped bank.
  ///
  /// The banks are the inner loop. They are independent of one another, so
  /// getting a command to all of them before starting the next is what turns
  /// that independence into overlapped activations.
  void emitAcrossBanks(OpBuilder &builder, Location loc, bool isAp,
                       const RowLoc &dst, const RowLoc &src, IRMapping &map) {
    Value lb = builder.create<arith::ConstantIndexOp>(loc, bankBase);
    Value ub =
        builder.create<arith::ConstantIndexOp>(loc, bankBase + bankCount);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    builder.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
          BankCoord coord = decomposeBank(b, bodyLoc, iv);
          Value dstAddr = materialise(b, bodyLoc, dst, coord, map);
          if (isAp) {
            b.create<pud::APOp>(bodyLoc, dstAddr, pud::Backend::Simdram);
          } else {
            Value srcAddr = materialise(b, bodyLoc, src, coord, map);
            b.create<pud::AAPOp>(bodyLoc, srcAddr, dstAddr,
                                 pud::Backend::Simdram);
          }
          b.create<scf::YieldOp>(bodyLoc);
        });
  }

  /// Instantiates `tmpl` with its data operands bound to `inputs` / `outputs`.
  LogicalResult emitTemplate(OpBuilder &builder, Location loc,
                             const RowOpTemplate &tmpl, ArrayRef<RowLoc> inputs,
                             ArrayRef<RowLoc> outputs, IRMapping &map) {
    for (const RowOpInstr &instr : tmpl.instrs) {
      RowLoc dst = resolve(instr.dst, inputs, outputs);
      RowLoc src = instr.isAp ? RowLoc() : resolve(instr.src, inputs, outputs);
      if (!dst.valid || (!instr.isAp && !src.valid))
        return failure();
      emitAcrossBanks(builder, loc, instr.isAp, dst, src, map);
      if (unresolvedIndex)
        return failure();
    }
    return success();
  }

private:
  struct BankCoord {
    Value channel;
    Value rank;
    Value bank;
    Value subarray;
  };

  /// Splits a flat bank index into that coordinate. The mapping numbers banks
  /// flatly and a channel holds a fixed number of them, so the index divides on
  /// that boundary.
  BankCoord decomposeBank(OpBuilder &b, Location loc, Value iv) {
    Value flat = b.create<arith::IndexCastOp>(loc, b.getI64Type(), iv);
    Value perChannel =
        b.create<arith::ConstantIntOp>(loc, geometry.banksPerChannel, 64);
    BankCoord coord;
    coord.channel = b.create<arith::DivSIOp>(loc, flat, perChannel);
    coord.bank = b.create<arith::RemSIOp>(loc, flat, perChannel);
    // Ranks and subarrays are part of the coordinate; nothing distributes over
    // them yet, so every address takes the first of each.
    coord.rank = b.create<arith::ConstantIntOp>(loc, 0, 64);
    coord.subarray = b.create<arith::ConstantIntOp>(loc, 0, 64);
    return coord;
  }

  Value materialise(OpBuilder &b, Location loc, const RowLoc &ref,
                    const BankCoord &coord, IRMapping &map) {
    if (ref.isControl)
      return b.create<pud::GetCtrlRowOp>(
          loc, pud::RowType::get(b.getContext(), groupOf(ref.control)),
          coord.channel, coord.rank, coord.bank, coord.subarray,
          toDialect(ref.control));

    Value row = b.create<arith::ConstantIntOp>(loc, ref.base + ref.offset, 64);
    if (ref.index) {
      // The index has to have been brought across already: what is being built
      // replaces the IR the original index belongs to, so referring to it would
      // outlive it.
      Value index = map.lookupOrNull(ref.index);
      if (!index)
        unresolvedIndex = true;
      else
        row = b.create<arith::AddIOp>(loc, row, index);
    }
    return b.create<pud::GetRowOp>(
        loc, pud::RowType::get(b.getContext(), kDataGroup), coord.channel,
        coord.rank, coord.bank, coord.subarray, row);
  }

  RowLoc resolve(const RowOperand &operand, ArrayRef<RowLoc> inputs,
                 ArrayRef<RowLoc> outputs) {
    switch (operand.kind) {
    case RowOperand::Kind::Control:
      return RowLoc::controlRow(operand.getControlRow());
    case RowOperand::Kind::Input:
      return operand.index < (int)inputs.size() ? inputs[operand.index]
                                                : RowLoc();
    case RowOperand::Kind::Output:
      return operand.index < (int)outputs.size() ? outputs[operand.index]
                                                 : RowLoc();
    case RowOperand::Kind::Temp:
      return binder.tempRow(operand.index);
    }
    return RowLoc();
  }

  /// The row type's groups: rows held at a constant value, rows that take part
  /// in computation, and rows that hold operands.
  static constexpr int64_t kComputeGroup = 0;
  static constexpr int64_t kConstantGroup = 1;
  static constexpr int64_t kDataGroup = 2;

  static int64_t groupOf(ControlRow row) {
    return (row == ControlRow::C_0 || row == ControlRow::C_1) ? kConstantGroup
                                                             : kComputeGroup;
  }

  static pud::SimdramCtrlRow toDialect(ControlRow row) {
    switch (row) {
    case ControlRow::T0: return pud::SimdramCtrlRow::T0;
    case ControlRow::T1: return pud::SimdramCtrlRow::T1;
    case ControlRow::T2: return pud::SimdramCtrlRow::T2;
    case ControlRow::T3: return pud::SimdramCtrlRow::T3;
    case ControlRow::DCC0: return pud::SimdramCtrlRow::DCC0;
    case ControlRow::DCC0N: return pud::SimdramCtrlRow::DCC0N;
    case ControlRow::DCC1: return pud::SimdramCtrlRow::DCC1;
    case ControlRow::DCC1N: return pud::SimdramCtrlRow::DCC1N;
    case ControlRow::DCC0N_T0: return pud::SimdramCtrlRow::DCC0N_T0;
    case ControlRow::DCC1N_T1: return pud::SimdramCtrlRow::DCC1N_T1;
    case ControlRow::T2_T3: return pud::SimdramCtrlRow::T2_T3;
    case ControlRow::T0_T3: return pud::SimdramCtrlRow::T0_T3;
    case ControlRow::T0_T1_T2: return pud::SimdramCtrlRow::T0_T1_T2;
    case ControlRow::T0_T1_T3: return pud::SimdramCtrlRow::T0_T1_T3;
    case ControlRow::DCC0_T1_T3: return pud::SimdramCtrlRow::DCC0_T1_T3;
    case ControlRow::DCC1_T0_T2: return pud::SimdramCtrlRow::DCC1_T0_T2;
    case ControlRow::C_0: return pud::SimdramCtrlRow::C_0;
    case ControlRow::C_1: return pud::SimdramCtrlRow::C_1;
    case ControlRow::NUM_CONTROL_ROWS: break;
    }
    llvm_unreachable("unhandled control row");
  }

  pud::MemoryGeometry geometry;
  RowBinder &binder;
  int64_t bankBase;
  int64_t bankCount;

public:
  /// Set when a row index named a value that was never brought across, which
  /// would have made an address out of the IR being replaced.
  bool unresolvedIndex = false;
};

//===----------------------------------------------------------------------===//
// Walking the mapped IR
//===----------------------------------------------------------------------===//

/// Which template a row op is an instance of, and how its operands bind to it.
struct TemplateUse {
  RowOpKind kind;
  SmallVector<RowLoc, 3> inputs;
  SmallVector<RowLoc, 2> outputs;
};

FailureOr<TemplateUse> describeRowOp(Operation *op, const RowBinder &binder) {
  auto rowOf = [&](Value v) -> FailureOr<RowLoc> {
    if (const RowLoc *loc = binder.rowFor(v))
      return *loc;
    return op->emitError() << "operand is not bound to any row";
  };

  TemplateUse use;
  if (auto add = dyn_cast<RowAddOp>(op)) {
    // The adder's inputs are (a, b, carry-in) and its outputs (sum, carry-out),
    // in the order the template's network declares them. A resident carry is in
    // the cell, so it is neither: the template names the cell directly.
    bool resident = hasResidentCarry(op);
    use.kind = resident ? RowOpKind::FullAdd : RowOpKind::FullAddLoaded;
    auto lhs = rowOf(add.getLhs()), rhs = rowOf(add.getRhs());
    auto sum = rowOf(add.getSum());
    if (failed(lhs) || failed(rhs) || failed(sum))
      return failure();
    use.inputs = {*lhs, *rhs};
    use.outputs = {*sum};
    if (!resident) {
      auto cin = rowOf(add.getCin()), cout = rowOf(add.getCout());
      if (failed(cin) || failed(cout))
        return failure();
      use.inputs.push_back(*cin);
      use.outputs.push_back(*cout);
    }
    return use;
  }

  if (isa<RowAndOp>(op))
    use.kind = RowOpKind::And;
  else if (isa<RowOrOp>(op))
    use.kind = RowOpKind::Or;
  else if (isa<RowXOrOp>(op))
    use.kind = RowOpKind::Xor;
  else if (isa<RowXNorOp>(op))
    use.kind = RowOpKind::Xnor;
  else if (isa<NotOp>(op))
    use.kind = RowOpKind::Not;
  else if (isa<MajOp>(op))
    use.kind = RowOpKind::Maj3;
  else
    return op->emitError() << "no row-op template for this op";

  for (Value operand : op->getOperands()) {
    auto loc = rowOf(operand);
    if (failed(loc))
      return failure();
    use.inputs.push_back(*loc);
  }
  auto result = rowOf(op->getResult(0));
  if (failed(result))
    return failure();
  use.outputs.push_back(*result);
  return use;
}

/// Brings the index arithmetic an address is built from into the new body.
///
/// A template writes its result where the value's consumer says it goes, and
/// that consumer can sit later in the block than the row op does: the index is
/// wanted where the command is issued, not where the IR being replaced happened
/// to compute it. It is pure arithmetic on loop variables and constants, so
/// computing it earlier is only ever computing it earlier.
LogicalResult materialiseIndex(OpBuilder &builder, Value v, IRMapping &map) {
  if (map.lookupOrNull(v))
    return success();
  Operation *def = v.getDefiningOp();
  if (!def || !isMemoryEffectFree(def))
    return failure();
  for (Value operand : def->getOperands())
    if (failed(materialiseIndex(builder, operand, map)))
      return failure();
  builder.clone(*def, map);
  return success();
}

/// Rebuilds a region as the row ops it issues, in the order it issues them.
class ScheduleEmitter {
public:
  ScheduleEmitter(RowOpEmitter &emitter, RowBinder &binder, PudBackend backend)
      : emitter(emitter), binder(binder), backend(backend) {}

  LogicalResult emit(Block &src, OpBuilder &builder, IRMapping &map);

private:
  RowOpEmitter &emitter;
  RowBinder &binder;
  PudBackend backend;
};

LogicalResult ScheduleEmitter::emit(Block &src, OpBuilder &builder,
                                    IRMapping &map) {
  for (Operation &op : src) {
    Location loc = op.getLoc();

    if (auto loop = dyn_cast<scf::ForOp>(&op)) {
      // A value the loop keeps in one place has to be put there first.
      for (const Seed &seed : binder.seedsFor(loop.getOperation())) {
        for (const RowLoc *r : {&seed.dst, &seed.src})
          if (r->index && failed(materialiseIndex(builder, r->index, map)))
            return op.emitError() << "the row a loop starts its carry from is "
                                     "at an index that cannot be computed here";
        emitter.emitAcrossBanks(builder, loc, /*isAp=*/false, seed.dst,
                                seed.src, map);
      }

      Value lb = map.lookupOrDefault(loop.getLowerBound());
      Value ub = map.lookupOrDefault(loop.getUpperBound());
      Value step = map.lookupOrDefault(loop.getStep());
      auto newLoop = builder.create<scf::ForOp>(loc, lb, ub, step);
      map.map(loop.getInductionVar(), newLoop.getInductionVar());

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(newLoop.getBody());
      if (failed(emit(*loop.getBody(), builder, map)))
        return failure();
      continue;
    }

    // A value that is not already where the slice wants it has to be put
    // there. Most of the time it already is, because a template writes its
    // result straight into the slot its consumer names; what is left is the
    // cases where it cannot, such as a carry that lives in a dual-contact cell
    // and has to be kept as a result row before the next addition clobbers it.
    if (auto insert = dyn_cast<InsertRowOp>(&op)) {
      const RowLoc *row = binder.rowFor(insert.getRow());
      const SliceLoc *slice = binder.sliceFor(insert.getSlice());
      if (!row || !slice)
        return op.emitError() << "insert of a row into a slice, one of which "
                                 "is not bound to any row";
      const RowLoc dst = slice->at(insert.getIndex());
      if (!row->sameAs(dst)) {
        for (const RowLoc *r : {&dst, row})
          if (r->index && failed(materialiseIndex(builder, r->index, map)))
            return op.emitError() << "the row this insert names is at an index "
                                     "that cannot be computed here";
        emitter.emitAcrossBanks(builder, loc, /*isAp=*/false, dst, *row, map);
      }
      continue;
    }

    if (isRowOp(&op)) {
      FailureOr<TemplateUse> use = describeRowOp(&op, binder);
      if (failed(use))
        return failure();
      auto bringIndices = [&](ArrayRef<RowLoc> group) {
        for (const RowLoc &row : group)
          if (row.index && failed(materialiseIndex(builder, row.index, map)))
            return failure();
        return success();
      };
      if (failed(bringIndices(use->inputs)) ||
          failed(bringIndices(use->outputs)))
        return op.emitError() << "a row this op names is at an index that "
                                 "cannot be computed where the command is "
                                 "issued";

      const RowOpTemplate *tmpl =
          RowOpLibrary::get().getTemplate(backend, use->kind);
      if (!tmpl)
        return op.emitError()
               << "no " << getRowOpName(use->kind) << " template for backend '"
               << getBackendName(backend)
               << "'; lime has no architecture definition for it";
      if (failed(emitter.emitTemplate(builder, loc, *tmpl, use->inputs,
                                      use->outputs, map)))
        return op.emitError() << "could not instantiate the "
                              << getRowOpName(use->kind) << " template";
      continue;
    }

    // The index arithmetic a row address is built from has to come along; the
    // dataflow that described which rows to touch does not, because the
    // addresses now say it.
    if (isa<arith::ArithDialect>(op.getDialect())) {
      builder.clone(op, map);
      continue;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Reading the mapping off the IR
//===----------------------------------------------------------------------===//

/// The loop the mapping annotated, and what it said.
struct MappedLoop {
  scf::ForOp loop;
  int64_t bankBase = 0;
  int64_t bankCount = 1;
  int64_t itersPerBank = 1;
};

/// How the memory the mapping chose banks in is organised, as it recorded it.
FailureOr<pud::MemoryGeometry> readGeometry(func::FuncOp func) {
  auto module = func->getParentOfType<ModuleOp>();
  auto dict = module ? module->getAttrOfType<DictionaryAttr>(
                           "bits.memory_geometry")
                     : DictionaryAttr();
  if (!dict) {
    func.emitError() << "no memory geometry on the module; run "
                        "bits-opt-mapping first";
    return failure();
  }
  pud::MemoryGeometry geometry;
  auto get = [&](StringRef name, int64_t &into) -> bool {
    auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(name));
    if (!attr)
      return false;
    into = attr.getInt();
    return true;
  };
  if (!get("total_banks", geometry.totalBanks) ||
      !get("banks_per_channel", geometry.banksPerChannel) ||
      !get("ranks_per_channel", geometry.ranksPerChannel)) {
    func.emitError() << "memory geometry is missing a field";
    return failure();
  }
  if (geometry.banksPerChannel <= 0) {
    func.emitError() << "a channel must hold at least one bank";
    return failure();
  }
  return geometry;
}

/// Slice-level ops `bits-pattern-apply` has no row expansion for yet. They are
/// what stands between a kernel and this pass, so they are worth naming.
bool isUnexpandedSliceOp(Operation *op) {
  return isa<AddIOp, AddIFullOp, SubIOp, MulIOp, AndOp, OrOp, XOrOp, XNOrOp,
             RangeScanOp, MaxOp, MinOp, MuxOp>(op);
}

FailureOr<MappedLoop> readMappedLoop(func::FuncOp func) {
  SmallVector<Operation *> rowOps;
  SmallVector<Operation *> sliceOps;
  func.walk([&](Operation *op) {
    if (isRowOp(op))
      rowOps.push_back(op);
    else if (isUnexpandedSliceOp(op))
      sliceOps.push_back(op);
  });

  // Say what is actually in the way. A kernel arrives here in one of three
  // states, and only one of them means bits-pattern-apply has not been run.
  if (!sliceOps.empty()) {
    llvm::SmallDenseSet<StringRef> names;
    for (Operation *op : sliceOps)
      names.insert(op->getName().getStringRef());
    auto diag = sliceOps.front()->emitError()
                << "bits-pattern-apply left slice-level ops with no row "
                   "expansion, so there is no row-level schedule to lower:";
    for (StringRef name : names)
      diag << " " << name;
    return failure();
  }
  if (rowOps.empty()) {
    func.emitError() << "no row-level Bits ops to lower; run "
                        "bits-pattern-apply first";
    return failure();
  }

  // The bank mapping sits on an enclosing loop; take the outermost that has it,
  // because that is the one whose iterations are the tiles.
  MappedLoop mapped;
  bool found = false;
  for (Operation *parent = rowOps.front(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    auto pf = loop->getAttrOfType<IntegerAttr>("bits.parallel_factor");
    if (!pf)
      continue;
    mapped.loop = loop;
    mapped.bankCount = pf.getInt();
    mapped.bankBase = 0;
    mapped.itersPerBank = 1;
    if (auto bid = loop->getAttrOfType<IntegerAttr>("bits.bank_id"))
      mapped.bankBase = bid.getInt();
    if (auto ipb = loop->getAttrOfType<IntegerAttr>("bits.iters_per_bank"))
      mapped.itersPerBank = ipb.getInt();
    found = true;
  }
  if (!found) {
    rowOps.front()->emitError() << "no bank mapping on any enclosing loop; run "
                                  "bits-opt-mapping first";
    return failure();
  }
  return mapped;
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

    auto geometry = readGeometry(func);
    if (failed(geometry))
      return signalPassFailure();

    auto mapped = readMappedLoop(func);
    if (failed(mapped))
      return signalPassFailure();

    // Every bank running this computation holds its operands in the same rows,
    // so the layout is decided once and the bank left open.
    pud::AddressAllocator allocator(*geometry);
    RowBinder binder(allocator, mapped->bankBase);
    if (failed(binder.bind(*mapped->loop.getBody())))
      return signalPassFailure();

    RowOpEmitter emitter(*geometry, binder, mapped->bankBase,
                         mapped->bankCount);
    ScheduleEmitter schedule(emitter, binder, target);

    // The new body is built beside the old one and swapped in at the end. It is
    // built from the old one -- the loop nest, the index arithmetic, the order
    // -- so the old one has to still be there while it is being read.
    Location loc = func.getLoc();
    auto entry = std::make_unique<Block>();
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(entry.get());

    IRMapping map;
    // Row indices are built from constants defined outside the mapped loop, so
    // bring those across before anything asks for them.
    for (Operation &op : *mapped->loop->getBlock())
      if (isa<arith::ArithDialect>(op.getDialect()))
        builder.clone(op, map);

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value tiles =
        builder.create<arith::ConstantIndexOp>(loc, mapped->itersPerBank);

    // The mapped loop ran once per tile of the whole operand; each bank takes
    // its own share of them, and the banks run at the same time.
    auto tileLoop = builder.create<scf::ForOp>(loc, zero, tiles, one);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(tileLoop.getBody());
      if (failed(schedule.emit(*mapped->loop.getBody(), builder, map)))
        return signalPassFailure();
    }

    builder.setInsertionPointToEnd(entry.get());
    builder.create<func::ReturnOp>(loc);

    Region &body = func.getBody();
    body.getBlocks().clear();
    body.push_back(entry.release());
    func.setType(FunctionType::get(func.getContext(), {}, {}));

    // The shape of the row space these addresses were handed out in. Whatever
    // realises them has to provide exactly this many rows per bank, and has to
    // split a flat bank index into a channel and a bank on the same boundary,
    // so the shape travels with the IR rather than being restated downstream.
    if (auto module = func->getParentOfType<ModuleOp>()) {
      module->setAttr("pud.data_rows_per_bank",
                      builder.getI64IntegerAttr(
                          allocator.rowsAllocated(mapped->bankBase)));
      module->setAttr("pud.banks_per_channel",
                      builder.getI64IntegerAttr(geometry->banksPerChannel));
      module->setAttr("pud.total_banks",
                      builder.getI64IntegerAttr(geometry->totalBanks));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::bits::createConvertBitsRowsToPuDPass() {
  return std::make_unique<ConvertBitsRowsToPuD>();
}
