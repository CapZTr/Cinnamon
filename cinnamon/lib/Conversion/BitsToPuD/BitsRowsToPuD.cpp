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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

/// What put the values a tensor holds into the machine.
///
/// A kernel is several nests one after another and a later one reads what an
/// earlier one wrote: the conv writes a run, `bits.assemble` presents it as a
/// tensor again, and the ReLU nest picks that tensor up with a
/// `tensor.extract_slice` and a `bits.transpose`. Between those two the values
/// never leave the machine, so the transpose is not a load -- it names rows
/// that already hold what it wants, and the only question is which ones.
enum class RowSource {
  /// It comes from outside the kernel, so the runtime is what lays it out and
  /// wherever the binder puts it is where it will be.
  External,
  /// A nest of this function wrote it, into the run this says.
  Produced,
  /// Nothing in the function accounts for it.
  Unknown,
};

/// The answer to "where is this tensor", with the run when there is one.
struct TensorOrigin {
  RowSource kind = RowSource::Unknown;
  SliceLoc run;
  /// How many rows the producer's run has, so a reader that wants more of them
  /// than were written can be told rather than left to read past the end.
  int64_t rows = 0;
};

/// How a run's rows move as the schedule works through the operand.
///
/// A nest runs its schedule over and over -- once per wave of the mapped loop,
/// and the whole of that once per repeat of the enclosing loops that did not
/// fit in the machine at once. Every one of those passes computes on a
/// different tile, so a run holding one pass's values has to be somewhere
/// different each time, or the pass after it writes over what the pass before
/// it left.
enum class Advance {
  /// One set of rows for the whole kernel. Scratch a template writes and reads
  /// back within the one command sequence, and an operand every pass reads the
  /// same values out of -- a broadcast of something an enclosing loop does not
  /// index.
  Fixed,
  /// A set per pass of the enclosing loops. An accumulator: every wave of a
  /// pass folds into the same rows, which is what makes it an accumulator, and
  /// the next pass starts a new sum somewhere else.
  PerPass,
  /// A set per wave, and per pass. An operand the mapped loop indexes, and the
  /// result of a nest that leaves one tile per wave.
  PerWave,
};

/// A run of rows, and how many copies of it the kernel was given.
struct RunInfo {
  int64_t rows = 0;
  int64_t copies = 1;
  Advance advance = Advance::Fixed;
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

  /// Says whether the nest about to be bound runs any wave that folds.
  ///
  /// A reduction whose every element already sits in a bank of its own -- the
  /// selection reading one distance per bank -- runs no waves at all: the whole
  /// of it is the cross-bank tree. That changes which of the fold's two operands
  /// is the earlier nest's run. With waves, the addend is what a wave reads and
  /// the accumulator is private; without them, the accumulator IS this bank's
  /// element and the addend only ever arrives in the tree's staging run.
  void beginNest(scf::ForOp mappedLoop, int64_t waves, int64_t repeat,
                 bool hasFoldingWaves, bool operandPerWave) {
    foldingWaves = hasFoldingWaves;
    nestWaves = std::max<int64_t>(1, waves);
    nestRepeat = std::max<int64_t>(1, repeat);
    plan.clear();
    if (!mappedLoop)
      return;

    // The induction variables a run's contents can depend on. The mapped loop's
    // is the wave; one further out hands this schedule a bank for each of its
    // iterations that fit at once and a pass for the rest, and the bank is free
    // -- every bank holds its own tile at the same row -- so what is left of it
    // is the pass.
    Value wave = mappedLoop.getInductionVar();
    SmallVector<Value> passes;
    for (Operation *parent = mappedLoop->getParentOp(); parent;
         parent = parent->getParentOp())
      if (auto parentFor = dyn_cast<scf::ForOp>(parent))
        passes.push_back(parentFor.getInductionVar());

    // What the waves of a pass read, and where what they produce goes. With the
    // operand resident each wave has its own rows of it and computes the answer
    // the kernel is for; without, the waves of a pass share one set and compute
    // on the same tile as many times as there are waves. That second mode is
    // not a smaller version of the first -- it is a model of what the schedule
    // costs, kept for the kernels whose operand does not fit in a subarray at
    // any bank count, where the alternative is nothing at all.
    Advance perWave = operandPerWave ? Advance::PerWave : Advance::PerPass;
    mappedLoop.getBody()->walk([&](TransposeOp transpose) {
      Value from = transpose.getInput();
      if (dependsOn(from, wave))
        plan[transpose.getOutput()] = perWave;
      else if (llvm::any_of(passes,
                            [&](Value iv) { return dependsOn(from, iv); }))
        plan[transpose.getOutput()] = Advance::PerPass;
    });

    // What the mapped loop carries is the accumulator, and every wave of a pass
    // folds into it on purpose.
    for (Value init : mappedLoop.getInits())
      if (isa<SliceType>(init.getType()))
        plan[init] = Advance::PerPass;

    // What the nest hands on. A reduction hands on its accumulator, which is
    // already spoken for; anything else writes one tile per wave and needs
    // somewhere to put each of them.
    mappedLoop.getBody()->walk([&](tensor::InsertOp insert) {
      if (Value run = runWrittenBy(insert.getScalar()))
        if (!plan.count(run))
          plan[run] = perWave;
    });
  }

  /// Binds what the mapped loop carries from one tile to the next.
  ///
  /// Its iter args are the values that outlive a tile -- the accumulator a
  /// reduction folds into. They are defined outside the body this pass rebuilds,
  /// so the walk over that body never reaches them and nothing else would give
  /// them a home. Every tile a bank runs folds into the same rows, which is what
  /// makes the accumulator an accumulator, so one run per bank is all it needs.
  LogicalResult bindCarried(scf::ForOp loop) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [i, init] : llvm::enumerate(loop.getInits())) {
      Value arg = loop.getRegionIterArg(i);
      Value result = loop.getResult(i);
      if (auto slice = dyn_cast<SliceType>(arg.getType())) {
        // What the accumulator starts from can be an earlier nest's result --
        // the selection starts from the first of the distances the nest before
        // it computed. Those rows are already in the machine, so the
        // accumulator can be that run rather than a copy of it, but only if it
        // is the last thing to want them: an accumulator is written every wave,
        // and a run something still has to read is not somewhere to write.
        auto produced = producedRunFor(init, slice.getBitWidth());
        if (failed(produced))
          return failure();
        if (*produced && foldingWaves &&
            readsRun(*loop.getBody(), (*produced)->base))
          produced = std::optional<SliceLoc>();
        SliceLoc loc = *produced ? **produced : newRun(init, slice.getBitWidth());
        slices[init] = loc;
        slices[arg] = loc;
        slices[result] = loc;
        // What a tile hands to the next one has to be where the next one will
        // look for it. The expansion writes its result into a run of its own,
        // and left to itself the binder would give that run somewhere else --
        // so every tile would fold into rows the following tile never reads,
        // and the accumulator would only ever hold the last tile's work. Follow
        // what the yield hands back to the run being written and give it the
        // same home. The trail runs through the bit loop the expansion built as
        // well as through its inserts: the fold is a loop over rows, so what
        // reaches the yield is that loop's result, and the run is the slice it
        // started from.
        if (Value carried = runWrittenBy(yield.getOperand(i)))
          slices[carried] = loc;
        continue;
      }
      if (isa<BitRowType>(arg.getType())) {
        RowLoc loc = scratchRow();
        rows[init] = loc;
        rows[arg] = loc;
        rows[result] = loc;
      }
      // Anything else -- a tensor of slices, say -- carries no rows of its own:
      // the slices it holds are bound where they are created.
    }
    return success();
  }

  const RowLoc *rowFor(Value v) const {
    auto it = rows.find(v);
    return it == rows.end() ? nullptr : &it->second;
  }
  const SliceLoc *sliceFor(Value v) const {
    auto it = slices.find(v);
    return it == slices.end() ? nullptr : &it->second;
  }
  /// Whether a run holds something loaded from outside the kernel, which is to
  /// say that the runtime has already put it there.
  bool isExternal(int64_t base) const { return external.contains(base); }

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

  /// A run of rows for something the body did not ask for -- the staging run a
  /// cross-bank fold copies its neighbour's partial sum into.
  SliceLoc allocateRun(int64_t numRows) { return newRun({}, numRows); }

  /// The shape of a run: how long it is, how many copies of it the kernel was
  /// given, and what steps between them.
  const RunInfo *runInfo(int64_t base) const {
    auto it = runs.find(base);
    return it == runs.end() ? nullptr : &it->second;
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
  SliceLoc newRun(Value forValue, int64_t numRows) {
    Advance advance = Advance::Fixed;
    if (forValue) {
      auto it = plan.find(forValue);
      if (it != plan.end())
        advance = it->second;
    }
    int64_t copies = advance == Advance::PerWave ? nestWaves * nestRepeat
                     : advance == Advance::PerPass ? nestRepeat
                                                   : 1;
    // A nest that runs its schedule once has nothing to step through. Saying so
    // is what keeps a run with a single copy out of the way of everything that
    // asks which pass it is on -- the reduce tree above all, which runs when
    // the waves are over and has no wave to be on.
    if (copies <= 1)
      advance = Advance::Fixed;
    SliceLoc loc =
        SliceLoc::run(allocator.allocate(flatBank, numRows * copies).row);
    runs[loc.base] = RunInfo{numRows, copies, advance};
    return loc;
  }

  /// Whether what a value holds is decided by an induction variable, following
  /// the index arithmetic and the tensor ops back. This is what says whether a
  /// wave reads its own tile of an operand or the same one every time.
  static bool dependsOn(Value value, Value iv) {
    llvm::SmallDenseSet<Value> seen;
    SmallVector<Value> worklist{value};
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!v || !seen.insert(v).second)
        continue;
      if (v == iv)
        return true;
      // A block argument other than the one being looked for is where a loop
      // hands something on, and what it hands on is its init.
      if (auto arg = dyn_cast<BlockArgument>(v)) {
        auto owner = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        if (owner && arg != owner.getInductionVar())
          worklist.push_back(owner.getInits()[arg.getArgNumber() - 1]);
        continue;
      }
      Operation *def = v.getDefiningOp();
      if (!def || seen.size() > 256)
        continue;
      for (Value operand : def->getOperands())
        worklist.push_back(operand);
    }
    return false;
  }

  /// Where the values a tensor holds already are, and what put them there.
  TensorOrigin traceTensor(Value tensor) const;

  /// The run an earlier nest left a slice's contents in, if it is read out of
  /// one -- null when the contents come from outside the kernel instead.
  ///
  /// Fails when a producer is found whose run is too short for what is being
  /// read out of it: that is two nests disagreeing about a layout, and quietly
  /// reading past the end of the run would be the worst way to find out.
  FailureOr<std::optional<SliceLoc>> producedRunFor(Value slice,
                                                    int64_t wantRows) const;

  /// The `bits.create_slice` whose run a value ends up being, following the
  /// inserts and the bit loops that wrote it. Null when the trail leads
  /// somewhere else, which means the run is not this loop's to place.
  static Value runWrittenBy(Value carried);

  /// Whether anything in a nest reads a run that is already in the machine --
  /// which is what makes it unsafe to write over.
  bool readsRun(Block &body, int64_t base) const;

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
  /// Runs holding an operand the runtime lays out. Nothing in the function
  /// writes them, which is the whole point of them.
  llvm::DenseSet<int64_t> external;
  /// What each run looks like, by the row it starts at.
  DenseMap<int64_t, RunInfo> runs;
  /// How the runs of the nest being bound move from pass to pass, for the
  /// values whose runs move at all.
  DenseMap<Value, Advance> plan;
  int64_t nestWaves = 1;
  int64_t nestRepeat = 1;
  bool foldingWaves = true;
};

/// Whether an op computes on rows, i.e. becomes a template instantiation.
bool isRowOp(Operation *op) {
  return isa<RowAddOp, RowSubOp, RowAndOp, RowOrOp, RowXOrOp, RowXNorOp,
             RowScanStepOp, RowScanFoldOp, RowGreaterStepOp, RowSelectOp,
             MajOp, NotOp>(op);
}

bool hasResidentCarry(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>(kCarryAttrName);
  return attr && attr.getValue() == kCarryResident;
}

/// The row a template keeps its loop-carried value in, and which of the op's
/// results is that value.
///
/// A carry and a borrow live in a dual-contact cell because the adder wants the
/// complement; a comparison's verdict lives in a compute row because the
/// comparator does not. Both are "the thing the loop hands to its next
/// iteration without a command", so both are described the same way here.
std::optional<std::pair<ControlRow, Value>> residentResult(Operation *op) {
  if (!hasResidentCarry(op))
    return std::nullopt;
  if (auto add = dyn_cast<RowAddOp>(op))
    return std::make_pair(ControlRow::DCC1, add.getCout());
  if (auto sub = dyn_cast<RowSubOp>(op))
    return std::make_pair(ControlRow::DCC1, sub.getBout());
  if (auto greater = dyn_cast<RowGreaterStepOp>(op))
    return std::make_pair(ControlRow::T2, greater.getGOut());
  return std::nullopt;
}

/// Follows a tensor back to whatever put its values where they are.
///
/// The walk is over the dataflow that is about to be discarded, and it is the
/// last thing that dataflow is good for: reshapes, slices and `bits.assemble`
/// hand on what they were given, a loop hands on what its yield holds, and a
/// `tensor.insert` of a slice is a nest saying "this run". An op the walk does
/// not know is external exactly when everything it reads is, which is what makes
/// a `linalg.broadcast` of a kernel operand an operand and not a mystery.
static TensorOrigin traceRun(Value v, const RowBinder &binder, unsigned depth) {
  TensorOrigin unknown;
  TensorOrigin external;
  external.kind = RowSource::External;
  // Deep enough for any chain a frontend builds, and a bound on a walk that has
  // no other reason to stop.
  if (!v || depth > 64)
    return unknown;

  // A slice is the end of the walk: it is a run, if it has been given one.
  if (auto sliceType = dyn_cast<SliceType>(v.getType())) {
    const SliceLoc *loc = binder.sliceFor(v);
    if (!loc || loc->broadcast)
      return unknown;
    TensorOrigin origin;
    origin.kind = RowSource::Produced;
    origin.run = *loc;
    origin.rows = sliceType.getBitWidth();
    return origin;
  }

  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop)
      return external; // the function's own argument
    if (arg == loop.getInductionVar())
      return unknown;
    return traceRun(loop.getInits()[arg.getArgNumber() - 1], binder, depth + 1);
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return unknown;
  if (auto loop = dyn_cast<scf::ForOp>(def)) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    return traceRun(yield.getOperand(cast<OpResult>(v).getResultNumber()),
                    binder, depth + 1);
  }
  if (auto insert = dyn_cast<tensor::InsertOp>(def)) {
    // Every element of the tensor a nest builds is the same run in a different
    // bank -- the schedule is written once with the bank left open -- so the
    // one being inserted answers for all of them.
    TensorOrigin scalar = traceRun(insert.getScalar(), binder, depth + 1);
    if (scalar.kind == RowSource::Produced)
      return scalar;
    return traceRun(insert.getDest(), binder, depth + 1);
  }
  if (auto assemble = dyn_cast<AssembleOp>(def))
    return traceRun(assemble.getInput(), binder, depth + 1);
  if (isa<tensor::EmptyOp>(def))
    return external;

  TensorOrigin result = external;
  bool sawTensor = false;
  for (Value operand : def->getOperands()) {
    if (!isa<TensorType, SliceType>(operand.getType()))
      continue;
    sawTensor = true;
    TensorOrigin from = traceRun(operand, binder, depth + 1);
    if (from.kind == RowSource::Unknown)
      return unknown;
    if (from.kind == RowSource::External)
      continue;
    if (result.kind == RowSource::Produced &&
        result.run.base != from.run.base)
      return unknown; // two nests at once, and no way to say which
    result = from;
  }
  return sawTensor ? result : unknown;
}

TensorOrigin RowBinder::traceTensor(Value tensor) const {
  return traceRun(tensor, *this, 0);
}

Value RowBinder::runWrittenBy(Value carried) {
  // The trail is finite -- each step goes to a strictly earlier definition --
  // but it is walked before anything has been checked, so it gets a bound.
  for (unsigned step = 0; carried && step < 64; ++step) {
    if (carried.getDefiningOp<CreateSliceOp>())
      return carried;
    if (auto insert = carried.getDefiningOp<InsertRowOp>()) {
      carried = insert.getSlice();
      continue;
    }
    if (auto inner = carried.getDefiningOp<scf::ForOp>()) {
      auto innerYield = cast<scf::YieldOp>(inner.getBody()->getTerminator());
      carried = innerYield.getOperand(cast<OpResult>(carried).getResultNumber());
      continue;
    }
    if (auto arg = dyn_cast<BlockArgument>(carried)) {
      auto owner = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      if (owner && arg != owner.getInductionVar()) {
        carried = owner.getInits()[arg.getArgNumber() - 1];
        continue;
      }
    }
    break;
  }
  return {};
}

bool RowBinder::readsRun(Block &body, int64_t base) const {
  bool found = false;
  body.walk([&](TransposeOp transpose) {
    TensorOrigin origin = traceTensor(transpose.getInput());
    if (origin.kind == RowSource::Produced && origin.run.base == base)
      found = true;
  });
  return found;
}

FailureOr<std::optional<SliceLoc>>
RowBinder::producedRunFor(Value slice, int64_t wantRows) const {
  auto transpose = slice.getDefiningOp<TransposeOp>();
  if (!transpose)
    return std::optional<SliceLoc>();
  TensorOrigin origin = traceTensor(transpose.getInput());
  if (origin.kind != RowSource::Produced)
    return std::optional<SliceLoc>();
  if (origin.rows < wantRows)
    return transpose.emitError()
           << "reads " << wantRows << " rows of a run an earlier nest wrote "
           << origin.rows << " rows of; the two nests disagree about the layout";
  return std::optional<SliceLoc>(origin.run);
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
    Operation *producer = carried.getDefiningOp();
    std::optional<ControlRow> residentIn;
    if (producer) {
      if (auto resident = residentResult(producer))
        if (carried == resident->second)
          residentIn = resident->first;
    }

    RowLoc home = residentIn ? RowLoc::controlRow(*residentIn)
                             : (from->isControl ? scratchRow() : *from);
    if (!home.sameAs(*from))
      seeds[loop.getOperation()].push_back({home, *from});

    rows[arg] = home;
    rows[result] = home;
  }

  // A select reads the comparison's verdict, and its complement, out of a
  // dual-contact cell; the comparison left it in a compute row. Moving it is one
  // command for the whole loop, so it belongs here beside the carry seeds rather
  // than inside the body, where it would cost one per bit.
  for (Operation &inner : *loop.getBody()) {
    auto select = dyn_cast<RowSelectOp>(&inner);
    if (!select)
      continue;
    Value verdict = select.getG();
    if (loop.getBody()->getParent()->isAncestor(
            verdict.getParentRegion()) &&
        verdict.getParentRegion() == loop.getBody()->getParent())
      continue; // defined in this loop: nothing to hoist
    const RowLoc *from = rowFor(verdict);
    if (!from)
      return select.emitError()
             << "a select whose verdict is not bound to any row";
    RowLoc dst = RowLoc::controlRow(ControlRow::DCC0);
    if (!from->sameAs(dst))
      seeds[loop.getOperation()].push_back({dst, *from});
    break;
  }
  return success();
}

LogicalResult RowBinder::bindOp(Operation *op) {
  // --- slices ---------------------------------------------------------------
  if (auto transpose = dyn_cast<TransposeOp>(op)) {
    auto slice = dyn_cast<SliceType>(transpose.getOutput().getType());
    if (!slice)
      return op->emitError() << "only a transpose to a slice can be placed";
    // A transpose of an earlier nest's result is not a load: those rows are
    // where that nest left them, and reading a run of this one's own would read
    // rows nothing ever wrote.
    auto produced = producedRunFor(transpose.getOutput(), slice.getBitWidth());
    if (failed(produced))
      return failure();
    // In a nest that runs no folding wave nothing in the body is ever emitted,
    // and the one thing this transpose could be -- the fold's addend -- reaches
    // the adder in the staging run the tree copies into, not here. Leaving it a
    // run of its own is what keeps that alias distinct from the accumulator.
    if (*produced && foldingWaves) {
      slices[transpose.getOutput()] = **produced;
      return success();
    }
    SliceLoc loc = newRun(transpose.getOutput(), slice.getBitWidth());
    // Otherwise it comes from outside, and the runtime fills it in before the
    // kernel runs; nothing in the function writes it and nothing has to. A
    // tensor the walk could not account for gets a run too, and is left for the
    // check at the end to name.
    if (traceTensor(transpose.getInput()).kind == RowSource::External)
      external.insert(loc.base);
    slices[transpose.getOutput()] = loc;
    return success();
  }
  if (auto create = dyn_cast<CreateSliceOp>(op)) {
    // Unless it already has a home: the run a reduction accumulates into is the
    // one the loop carries, decided before the walk got here.
    if (!slices.count(create.getSlice())) {
      auto slice = cast<SliceType>(create.getSlice().getType());
      slices[create.getSlice()] = newRun(create.getSlice(), slice.getBitWidth());
    }
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
  if (auto ext = dyn_cast<ExtensionIOp>(op)) {
    // An extension has no rows of its own. bits-pattern-apply folds it into the
    // bit loop of whatever computes on it -- the rows above the operand are a
    // row the substrate already holds -- so one that is still here and gets read
    // a row at a time is an expansion that was never written, and binding it to
    // the run underneath would quietly read the wrong rows past the operand's
    // width. One whose result only ever leaves the kernel as a value is fine and
    // is what the popcount and BNN kernels do with their final widening.
    for (OpOperand &use : ext.getResult().getUses())
      if (isa<ExtractRowOp, ExtractSubSliceOp, InsertRowOp>(use.getOwner()))
        return op->emitError()
               << "bits.ext_i reached row level and is read by '"
               << use.getOwner()->getName().getStringRef()
               << "'; the expansion that consumes it has to read the rows under "
                  "the extension itself";
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
  if (auto one = dyn_cast<CreateAllOneRowOp>(op)) {
    rows[one.getRow()] = RowLoc::controlRow(ControlRow::C_1);
    return success();
  }
  if (auto create = dyn_cast<CreateRowOp>(op)) {
    rows[create.getRow()] = scratchRow();
    return success();
  }
  if (auto extract = dyn_cast<ExtractRowOp>(op)) {
    const SliceLoc *from = sliceFor(extract.getSlice());
    if (!from)
      return op->emitError()
             << "extract from a slice that is not bound to any rows; it comes "
                "from "
             << (extract.getSlice().getDefiningOp()
                     ? extract.getSlice().getDefiningOp()->getName().getStringRef()
                     : llvm::StringRef("a block argument"));
    rows[extract.getRow()] = from->at(extract.getRowIdx());
    return success();
  }

  // --- row ops --------------------------------------------------------------
  if (isRowOp(op)) {
    if (auto greater = dyn_cast<RowGreaterStepOp>(op)) {
      rows[greater.getGOut()] = hasResidentCarry(op)
                                    ? RowLoc::controlRow(ControlRow::T2)
                                    : homeFor(greater.getGOut());
      return success();
    }
    if (auto step = dyn_cast<RowScanStepOp>(op)) {
      // Both chains stay in their cells for the whole scan; the template names
      // them, so neither result is ever placed anywhere else. The "<= upper"
      // chain is kept on its cell's complementing contact -- the step's two
      // majorities then read the same value rather than a value and its
      // complement, which no third cell was free to supply. See
      // buildRangeScanStep.
      rows[step.getGeOut()] = RowLoc::controlRow(ControlRow::DCC0);
      rows[step.getLeOut()] = RowLoc::controlRow(ControlRow::DCC1N);
      return success();
    }
    if (auto sub = dyn_cast<RowSubOp>(op)) {
      rows[sub.getDiff()] = homeFor(sub.getDiff());
      rows[sub.getBout()] = hasResidentCarry(op)
                                ? RowLoc::controlRow(ControlRow::DCC1)
                                : homeFor(sub.getBout());
      return success();
    }
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

/// Checks that every run a schedule reads is one something wrote.
///
/// The binder decides where each row-level value lives, and a run it hands out
/// that nothing ever writes is not a compile error anywhere else: the schedule
/// is well-formed, it is priced, it runs, and it computes on whatever the last
/// kernel left in those rows. The chains where this goes wrong are the ones
/// between nests -- the writer is one nest and the reader the next, with the
/// tensor dataflow the only thing connecting them -- which is exactly the part
/// this pass throws away. So it is checked before it is thrown away.
///
/// Three things count as writing a run: an operand of the kernel, which the
/// runtime lays out before the first command; an earlier nest, reached by
/// following the tensor back to it; and a `bits.insert_row` in this function.
LogicalResult verifyReadsHaveWriters(func::FuncOp func,
                                     const RowBinder &binder,
                                     const llvm::DenseSet<Operation *> &unemitted) {
  llvm::DenseSet<int64_t> written;
  func.walk([&](InsertRowOp insert) {
    if (const SliceLoc *loc = binder.sliceFor(insert.getSlice()))
      if (!loc->broadcast)
        written.insert(loc->base);
  });

  LogicalResult result = success();
  func.walk([&](ExtractRowOp extract) {
    // A fold that no wave runs is not emitted, so what it names is not read.
    for (Operation *parent = extract; parent; parent = parent->getParentOp())
      if (unemitted.contains(parent))
        return;
    const SliceLoc *loc = binder.sliceFor(extract.getSlice());
    // Unbound is a different complaint, and bindOp has already made it.
    if (!loc || loc->broadcast || binder.isExternal(loc->base) ||
        written.contains(loc->base))
      return;
    Operation *def = extract.getSlice().getDefiningOp();
    auto diag = extract.emitError()
                << "reads rows nothing writes: the run at row " << loc->base
                << " holds no operand of the kernel, no earlier nest's result "
                   "and nothing this function puts there";
    diag.attachNote(def ? def->getLoc() : extract.getLoc())
        << "the slice being read comes from "
        << (def ? def->getName().getStringRef() : StringRef("a block argument"));
    result = failure();
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Issues the commands of a template, one bank at a time.
class RowOpEmitter {
public:
  RowOpEmitter(const pud::MemoryGeometry &geometry, RowBinder &binder,
               ArrayRef<int64_t> banks)
      : geometry(geometry), binder(binder), allBanks(banks.begin(), banks.end()),
        banks(this->allBanks) {}

  /// Runs the rest of this scope's commands on `subset` of the mapped banks.
  ///
  /// A tree level folds into some of them and not others, and the level's
  /// reduce is one wave over exactly those.
  class BankScope {
  public:
    BankScope(RowOpEmitter &emitter, ArrayRef<int64_t> subset)
        : emitter(emitter), saved(emitter.banks) {
      emitter.banks.assign(subset.begin(), subset.end());
    }
    ~BankScope() { emitter.banks = saved; }

  private:
    RowOpEmitter &emitter;
    SmallVector<int64_t> saved;
  };

  /// Rows that stand in for others while a piece of the schedule is emitted.
  ///
  /// The same commands mean different rows in different places: the wave that
  /// seeds a reduction writes its product where the accumulator lives rather
  /// than into a run of its own, and a fold reads its addend from where the copy
  /// that feeds it put it. Both are the schedule saying which rows a command
  /// names, so both are a substitution over the binding rather than a second
  /// binding.
  void aliasRun(int64_t from, int64_t to) { runAlias[from] = to; }
  void clearAliases() { runAlias.clear(); }

  /// Which pass of the schedule the commands being emitted belong to.
  ///
  /// A run that holds one pass's values is a different run of rows each pass,
  /// so an address into it is the run's base plus the pass's share of it. Both
  /// indices are given because a run steps with one or the other: an operand
  /// the mapped loop indexes moves every wave, an accumulator only when the
  /// enclosing loops come round again. Null means the commands are outside any
  /// pass -- the reduce tree, which runs once when the waves are done.
  void setPass(Value wave, Value pass) {
    waveIndex = wave;
    passIndex = pass;
  }

  ArrayRef<int64_t> mappedBanks() const { return allBanks; }

  /// Issues one command to every mapped bank.
  ///
  /// The banks are the inner loop. They are independent of one another, so
  /// getting a command to all of them before starting the next is what turns
  /// that independence into overlapped activations.
  void emitAcrossBanks(OpBuilder &builder, Location loc, bool isAp,
                       const RowLoc &dst, const RowLoc &src, IRMapping &map) {
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    int64_t perChannel = std::max<int64_t>(1, geometry.banksPerChannel);
    for (auto [from, to] : bankRanges()) {
      // A range is within one channel, so the channel is a constant here and
      // the loop counts the banks of it. Counting flat banks instead and
      // dividing would put a division and a multiply by the channel's span in
      // the innermost loop of the kernel, on a machine wide enough for the
      // division not to fold -- which is every machine this is for.
      int64_t channel = from / perChannel;
      Value lb = builder.create<arith::ConstantIndexOp>(loc, from - channel * perChannel);
      Value ub = builder.create<arith::ConstantIndexOp>(loc, to - channel * perChannel);
      builder.create<scf::ForOp>(
          loc, lb, ub, step, ValueRange{},
          [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
            BankCoord coord = bankInChannel(b, bodyLoc, channel, iv);
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
  }

  /// Copies a run of rows along every one of `edges`, a row at a time.
  ///
  /// This is the one command that names two banks, and the only way a value a
  /// reduction spread over banks reaches the bank that folds it in.
  ///
  /// The rows are the outer loop and the edges the inner one, for the same
  /// reason the banks are the inner loop everywhere else: the pairs are
  /// independent of one another, and a copy occupies the two banks it names for
  /// a whole row's worth of transfer. Sending one row along every edge before
  /// starting the next row lets those transfers overlap; doing one edge's whole
  /// run first would leave every other bank waiting, and would hold two of them
  /// busy for long enough that the rank could not be refreshed in between.
  void emitCopies(OpBuilder &builder, Location loc,
                  ArrayRef<std::pair<int64_t, int64_t>> edges, int64_t srcRun,
                  int64_t dstRun, int64_t rows) {
    if (edges.empty() || rows <= 0)
      return;
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value count = builder.create<arith::ConstantIndexOp>(loc, rows);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    builder.create<scf::ForOp>(
        loc, zero, count, step, ValueRange{},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
          Value row =
              b.create<arith::IndexCastOp>(bodyLoc, b.getI64Type(), iv);
          auto rowIn = [&](int64_t bank, int64_t run) {
            BankCoord coord = decomposeBank(b, bodyLoc, bank);
            Value base = b.create<arith::ConstantIntOp>(bodyLoc, run, 64);
            return b.create<pud::GetRowOp>(
                bodyLoc, pud::RowType::get(b.getContext(), kDataGroup),
                coord.channel, coord.rank, coord.bank, coord.subarray,
                b.create<arith::AddIOp>(bodyLoc, base, row));
          };
          int64_t perChannel = std::max<int64_t>(1, geometry.banksPerChannel);
          int64_t total = std::max<int64_t>(1, geometry.totalBanks);
          auto channelOf = [&](int64_t bank) {
            return (((bank % total) + total) % total) / perChannel;
          };
          for (auto [srcBank, dstBank] : edges) {
            Value src = rowIn(srcBank, srcRun);
            Value dst = rowIn(dstBank, dstRun);
            // One controller owns both ends, or neither owns both. A copy
            // within a channel is the one command it looks like; across
            // channels neither controller can reach the other's row, so it is
            // the row out of one and the row into the other, each addressed to
            // the controller that owns it. This is the expansion the trace
            // player has always replayed a cross-channel copy as.
            if (channelOf(srcBank) == channelOf(dstBank)) {
              b.create<pud::RowCopyOp>(bodyLoc, src, dst,
                                       pud::Backend::Simdram);
            } else {
              b.create<pud::RowStreamOp>(bodyLoc, src, /*out=*/true,
                                         pud::Backend::Simdram);
              b.create<pud::RowStreamOp>(bodyLoc, dst, /*out=*/false,
                                         pud::Backend::Simdram);
            }
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

  /// The banks in scope, as ranges of flat bank indices.
  ///
  /// The mapping hands out one contiguous group per concurrent instance of the
  /// schedule, wrapping at the end of the machine exactly as its own
  /// `buildBankGroup` numbers them, so consecutive banks collapse back into the
  /// loops they came from: one range is one loop, and the common case of a
  /// single unwrapped group is the one loop it always was.
  SmallVector<std::pair<int64_t, int64_t>> bankRanges() const {
    int64_t total = std::max<int64_t>(1, geometry.totalBanks);
    SmallVector<int64_t> flat;
    flat.reserve(banks.size());
    for (int64_t bank : banks)
      flat.push_back(((bank % total) + total) % total);
    llvm::sort(flat);
    flat.erase(llvm::unique(flat), flat.end());

    SmallVector<std::pair<int64_t, int64_t>> ranges;
    int64_t perChannel = std::max<int64_t>(1, geometry.banksPerChannel);
    for (int64_t bank : flat) {
      // A range never crosses a channel: the loop that runs it takes its
      // channel as a constant, and one loop cannot have two.
      if (!ranges.empty() && ranges.back().second == bank &&
          ranges.back().first / perChannel == bank / perChannel)
        ranges.back().second = bank + 1;
      else
        ranges.push_back({bank, bank + 1});
    }
    return ranges;
  }

  /// A bank counted out within a channel that is known here.
  BankCoord bankInChannel(OpBuilder &b, Location loc, int64_t channel,
                          Value iv) {
    BankCoord coord;
    coord.channel = b.create<arith::ConstantIntOp>(loc, channel, 64);
    coord.bank = b.create<arith::IndexCastOp>(loc, b.getI64Type(), iv);
    // Ranks and subarrays are part of the coordinate; nothing distributes over
    // them yet, so every address takes the first of each.
    coord.rank = b.create<arith::ConstantIntOp>(loc, 0, 64);
    coord.subarray = b.create<arith::ConstantIntOp>(loc, 0, 64);
    return coord;
  }

  /// The same split for a bank that is known here rather than counted out by a
  /// loop -- the ends of a copy, which name one bank each.
  BankCoord decomposeBank(OpBuilder &b, Location loc, int64_t bank) {
    int64_t total = std::max<int64_t>(1, geometry.totalBanks);
    int64_t flat = ((bank % total) + total) % total;
    BankCoord coord;
    coord.channel = b.create<arith::ConstantIntOp>(
        loc, flat / geometry.banksPerChannel, 64);
    coord.bank = b.create<arith::ConstantIntOp>(
        loc, flat % geometry.banksPerChannel, 64);
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

    auto alias = runAlias.find(ref.base);
    int64_t base = alias == runAlias.end() ? ref.base : alias->second;
    Value row = b.create<arith::ConstantIntOp>(loc, base + ref.offset, 64);
    // A run that holds one pass's values has one set of rows per pass, laid out
    // end to end, so this pass's set is that many runs into it. The index is
    // the loop variable of whichever loop the run steps with, and the whole
    // term is loop-invariant inside the bank loop that issues the commands,
    // which is what keeps this arithmetic rather than a second address.
    if (const RunInfo *info = binder.runInfo(base)) {
      Value step = info->advance == Advance::PerWave    ? waveIndex
                   : info->advance == Advance::PerPass ? passIndex
                                                       : Value();
      if (!step && info->advance != Advance::Fixed)
        strayPassRow = true;
      if (step && info->rows > 0) {
        Value stride = b.create<arith::ConstantIntOp>(loc, info->rows, 64);
        row = b.create<arith::AddIOp>(
            loc, row, b.create<arith::MulIOp>(loc, step, stride));
      }
    }
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
  /// Every bank the schedule was mapped onto, and the ones a command reaches
  /// right now -- the same thing outside a tree level.
  SmallVector<int64_t> allBanks;
  SmallVector<int64_t> banks;
  DenseMap<int64_t, int64_t> runAlias;
  /// Where in the passes the commands being emitted are, as i64 values in the
  /// function being built.
  Value waveIndex;
  Value passIndex;

public:
  /// Set when a row index named a value that was never brought across, which
  /// would have made an address out of the IR being replaced.
  bool unresolvedIndex = false;
  /// Set when a run that holds one pass's values was addressed from somewhere
  /// with no pass to be on, which would have named the first pass's rows
  /// whatever pass the schedule was actually running.
  bool strayPassRow = false;
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

  if (auto sub = dyn_cast<RowSubOp>(op)) {
    // The subtractor exists only in the form that keeps its borrow in the cell.
    // A loaded-borrow variant would be a different template, and nothing builds
    // one, so say so rather than silently pricing this as the resident one.
    if (!hasResidentCarry(op))
      return op->emitError()
             << "a subtraction whose borrow comes from a data row has no "
                "template; only the resident-borrow form is expanded";
    use.kind = RowOpKind::FullSub;
    auto lhs = rowOf(sub.getLhs()), rhs = rowOf(sub.getRhs());
    auto diff = rowOf(sub.getDiff());
    if (failed(lhs) || failed(rhs) || failed(diff))
      return failure();
    use.inputs = {*lhs, *rhs};
    use.outputs = {*diff};
    return use;
  }

  if (auto greater = dyn_cast<RowGreaterStepOp>(op)) {
    if (!hasResidentCarry(op))
      return op->emitError()
             << "a comparison whose verdict comes from a data row has no "
                "template; only the resident-verdict form is expanded";
    use.kind = RowOpKind::GreaterStep;
    auto lhs = rowOf(greater.getLhs()), rhs = rowOf(greater.getRhs());
    if (failed(lhs) || failed(rhs))
      return failure();
    use.inputs = {*lhs, *rhs};
    return use;
  }
  if (auto select = dyn_cast<RowSelectOp>(op)) {
    use.kind = RowOpKind::Select;
    auto lhs = rowOf(select.getLhs()), rhs = rowOf(select.getRhs());
    auto result = rowOf(select.getResult());
    if (failed(lhs) || failed(rhs) || failed(result))
      return failure();
    // The verdict is read out of the cell the select's template names, so it is
    // not one of the operands the template is told about.
    use.inputs = {*lhs, *rhs};
    use.outputs = {*result};
    return use;
  }
  if (auto step = dyn_cast<RowScanStepOp>(op)) {
    use.kind = getRangeScanKind(step.getLowerBit(), step.getUpperBit());
    auto value = rowOf(step.getValue());
    if (failed(value))
      return failure();
    // The chains are in the cells the template reads directly, so the only
    // operand it needs told about is the column's bit plane.
    use.inputs = {*value};
    return use;
  }
  if (isa<RowScanFoldOp>(op)) {
    use.kind = RowOpKind::ScanFold;
    auto mask = rowOf(op->getResult(0));
    if (failed(mask))
      return failure();
    use.outputs = {*mask};
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
  /// Just these ops, wherever the builder is -- one piece of a body.
  LogicalResult emit(ArrayRef<Operation *> ops, OpBuilder &builder,
                     IRMapping &map);

  /// Ops to walk past without issuing anything.
  ///
  /// The wave that seeds a reduction does everything the others do except fold,
  /// because there is nothing yet to fold into: it writes where the accumulator
  /// lives instead. Leaving the fold out is the whole of that difference.
  void suppress(ArrayRef<Operation *> ops) {
    suppressed.insert(ops.begin(), ops.end());
  }
  void emitEverything() { suppressed.clear(); }

private:
  llvm::SmallPtrSet<Operation *, 4> suppressed;

  /// Puts a value the template is going to read from a named control row into
  /// that row, unless it is already there.
  ///
  /// This is what seeds a range scan's two chains: the first step's operands are
  /// the constant-one row, so it costs one copy each, and every step after reads
  /// what the previous one left in the cell, so it costs nothing. The count
  /// falls out of the schedule rather than being asserted.
  LogicalResult stageInto(OpBuilder &builder, Location loc, Value value,
                          ControlRow target, IRMapping &map) {
    const RowLoc *from = binder.rowFor(value);
    if (!from)
      return failure();
    RowLoc dst = RowLoc::controlRow(target);
    if (from->sameAs(dst))
      return success();
    if (from->index && failed(materialiseIndex(builder, from->index, map)))
      return failure();
    emitter.emitAcrossBanks(builder, loc, /*isAp=*/false, dst, *from, map);
    return success();
  }

  RowOpEmitter &emitter;
  RowBinder &binder;
  PudBackend backend;
};

LogicalResult ScheduleEmitter::emit(Block &src, OpBuilder &builder,
                                    IRMapping &map) {
  SmallVector<Operation *> ops;
  for (Operation &op : src)
    ops.push_back(&op);
  return emit(ops, builder, map);
}

LogicalResult ScheduleEmitter::emit(ArrayRef<Operation *> ops,
                                    OpBuilder &builder, IRMapping &map) {
  for (Operation *opPtr : ops) {
    Operation &op = *opPtr;
    Location loc = op.getLoc();
    if (suppressed.contains(&op))
      continue;

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

      // The bounds may not have been brought across yet: a piece of the body
      // can be emitted on its own, away from where the rest of it cloned its
      // constants, and then they have to come with it.
      for (Value bound : {loop.getLowerBound(), loop.getUpperBound(),
                          loop.getStep()})
        if (failed(materialiseIndex(builder, bound, map)))
          return op.emitError() << "a loop bound cannot be computed here";
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
      // A scan's chains are read straight out of the cells, so they have to be
      // in them before the sequence runs.
      Value ge, le;
      if (auto step = dyn_cast<RowScanStepOp>(&op)) {
        ge = step.getGe();
        le = step.getLe();
      } else if (auto fold = dyn_cast<RowScanFoldOp>(&op)) {
        ge = fold.getGe();
        le = fold.getLe();
      }
      // The "<= upper" chain is kept complemented, on its cell's other
      // contact, so that both of a step's majorities read the value rather than
      // one of them its complement -- staging it anywhere else would turn the
      // cell over once a bit for nothing. See buildRangeScanStep.
      if (ge && (failed(stageInto(builder, loc, ge, ControlRow::DCC0, map)) ||
                 failed(stageInto(builder, loc, le, ControlRow::DCC1N, map))))
        return op.emitError() << "a range scan's chain is not in a row the "
                                 "scan step can be given it from";

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
    //
    // Not all of it can: arithmetic on the mapped loop's own induction variable
    // says which tile the loop was on, and there is no such variable here -- the
    // waves are the tiles and every one of them uses the same rows. If a row
    // address turns out to need it after all, materialising that address says
    // so; what must not happen is bringing the arithmetic across still pointing
    // at the body being replaced.
    if (isa<arith::ArithDialect>(op.getDialect())) {
      if (llvm::none_of(op.getOperands(),
                        [&](Value v) { return !map.contains(v); }))
        builder.clone(op, map);
      continue;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Reading the mapping off the IR
//===----------------------------------------------------------------------===//

/// The loop the mapping placed, and where it put it.
///
/// The loop is the innermost one carrying a mapping, because that is the one
/// whose body is a wave: an enclosing annotated loop does not run its body in
/// sequence here, it hands out banks for this one to run its waves on at the
/// same time.
struct MappedLoop {
  scf::ForOp loop;
  /// Every bank the schedule runs on, in group order.
  ///
  /// One group of `parallel_factor` banks holds one instance of the schedule,
  /// and the enclosing loop's iterations put `concurrent_groups` of them side by
  /// side, `bank_group_stride` banks apart, wrapping at the end of the machine.
  SmallVector<int64_t> banks;
  /// Banks in one group: the tree's indices are within a group, and every group
  /// runs the same tree.
  int64_t banksPerGroup = 1;
  int64_t groupStride = 1;
  /// How many waves of the schedule each bank runs.
  ///
  /// A bank row is wider than one operand, so several tiles sit side by side in
  /// it. Nothing addresses a lane -- an activation drives the whole row -- so
  /// the commands for the tiles sharing a row are the SAME commands, and a bank
  /// runs one wave per group of them rather than one per tile.
  int64_t waves = 1;
  int64_t packFactor = 1;
  /// How many times over the enclosing loops make the whole thing run.
  int64_t repeat = 1;
  /// Waves that only produce, before the first one that folds into the
  /// accumulator. Zero when the expansion peeled the seed itself.
  int64_t seedWaves = 0;
  /// Levels of intra-bank folding of the packed lanes, then the cross-bank tree
  /// as lists of (source, destination) banks within a group.
  int64_t foldLevels = 0;
  /// Folds a bank does to bring together the elements an earlier nest left in
  /// it -- one per pass that nest took beyond the first. They come before the
  /// tree and read a run further into the same rows, which is where the pass
  /// that produced them put them.
  int64_t passFolds = 0;
  SmallVector<SmallVector<std::pair<int64_t, int64_t>>> tree;
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
      !get("ranks_per_channel", geometry.ranksPerChannel) ||
      !get("row_stride", geometry.rowStride)) {
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
  return isa<AddIOp, AddIFullOp, MulIOp, AndOp, OrOp, XOrOp, XNOrOp, MaxOp,
             MinOp, MuxOp>(op);
}

FailureOr<MappedLoop> readMappedLoop(scf::ForOp loop);

/// Everything a function asks the machine to do, in the order it asks.
///
/// A function can hold more than one nest. They are chained rather than run
/// together -- a ReLU consumes the convolution before it, a selection consumes
/// the distances before it -- which is how the mapping schedules them too, one
/// after another on a cursor that runs through the whole function.
FailureOr<SmallVector<MappedLoop>> readMappedLoops(func::FuncOp func) {
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

  // The mapping sits on an enclosing loop; for each row op take the INNERMOST
  // one that has it. That is the loop whose body is one wave. An annotated loop
  // further out is not a loop here at all: its iterations are independent, so
  // the mapping gave each of them its own group of banks to run this schedule
  // on at the same time, and what it contributes is banks, not iterations.
  SmallVector<scf::ForOp> loops;
  llvm::SmallDenseSet<Operation *> seen;
  for (Operation *rowOp : rowOps) {
    scf::ForOp loop;
    for (Operation *parent = rowOp; parent && !loop;
         parent = parent->getParentOp()) {
      auto candidate = dyn_cast<scf::ForOp>(parent);
      if (candidate && candidate->hasAttr("bits.parallel_factor"))
        loop = candidate;
    }
    if (!loop) {
      rowOp->emitError() << "no bank mapping on any enclosing loop; run "
                            "bits-opt-mapping first";
      return failure();
    }
    if (seen.insert(loop.getOperation()).second)
      loops.push_back(loop);
  }

  SmallVector<MappedLoop> mappedLoops;
  for (scf::ForOp loop : loops) {
    auto mapped = readMappedLoop(loop);
    if (failed(mapped))
      return failure();
    mappedLoops.push_back(std::move(*mapped));
  }
  return mappedLoops;
}

FailureOr<MappedLoop> readMappedLoop(scf::ForOp loop) {
  auto readAttr = [&](StringRef name, int64_t fallback) {
    auto attr = loop->getAttrOfType<IntegerAttr>(name);
    return attr ? std::max<int64_t>(1, attr.getInt()) : fallback;
  };

  MappedLoop mapped;
  mapped.loop = loop;
  mapped.banksPerGroup = readAttr("bits.parallel_factor", 1);
  mapped.groupStride = readAttr("bits.bank_group_stride", mapped.banksPerGroup);
  mapped.packFactor = readAttr("bits.pack_factor", 1);
  mapped.repeat = readAttr("bits.wave_repeat", 1);

  // The tree names the banks of one group, and having named them it is also
  // what says which banks the group is: a reduction's group is the banks its
  // tree folds, and only when there is no tree does the group run from the bank
  // the ILP chose.
  SmallVector<int64_t> group;
  if (auto treeAttr = loop->getAttrOfType<ArrayAttr>("bits.reduce_tree")) {
    llvm::SmallDenseSet<int64_t> seen;
    for (Attribute levelAttr : treeAttr) {
      auto level = dyn_cast<ArrayAttr>(levelAttr);
      if (!level)
        continue;
      SmallVector<std::pair<int64_t, int64_t>> edges;
      for (Attribute pairAttr : level) {
        auto pair = dyn_cast<DictionaryAttr>(pairAttr);
        if (!pair)
          continue;
        auto src = dyn_cast_or_null<IntegerAttr>(pair.get("src_bank"));
        auto dst = dyn_cast_or_null<IntegerAttr>(pair.get("dst_bank"));
        if (!src || !dst)
          continue;
        edges.push_back({src.getInt(), dst.getInt()});
        for (int64_t bank : {src.getInt(), dst.getInt()})
          if (seen.insert(bank).second)
            group.push_back(bank);
      }
      if (!edges.empty())
        mapped.tree.push_back(std::move(edges));
    }
    llvm::sort(group);
  }
  if (group.empty()) {
    int64_t base = 0;
    if (auto bid = loop->getAttrOfType<IntegerAttr>("bits.bank_id"))
      base = bid.getInt();
    for (int64_t i = 0; i < mapped.banksPerGroup; ++i)
      group.push_back(base + i);
  }
  mapped.banksPerGroup = group.size();

  int64_t groups = readAttr("bits.concurrent_groups", 1);
  for (int64_t g = 0; g < groups; ++g)
    for (int64_t bank : group)
      mapped.banks.push_back(bank + g * mapped.groupStride);

  // How many waves a bank runs: its share of the loop's iterations, and one
  // wave per group of them that shares a row. The share is taken from the trip
  // count rather than from `iters_per_bank`, which is the same number except
  // when the banks do not divide the iterations -- the annotation rounds that
  // down and would leave a wave's worth of work undone, where the cost model
  // rounds it up. 15 candidates over 4 banks is 4 waves, not 3.
  int64_t itersPerBank = readAttr("bits.iters_per_bank", 1);
  APInt lower, upper, step;
  if (matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) &&
      matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) &&
      matchPattern(loop.getStep(), m_ConstantInt(&step)) &&
      step.getSExtValue() > 0) {
    int64_t trip = (upper.getSExtValue() - lower.getSExtValue() +
                    step.getSExtValue() - 1) /
                   step.getSExtValue();
    if (trip > 0)
      itersPerBank = (trip + mapped.banksPerGroup - 1) / mapped.banksPerGroup;
  }
  mapped.waves = (itersPerBank + mapped.packFactor - 1) / mapped.packFactor;

  mapped.foldLevels =
      mapped.packFactor > 1 ? (int64_t)llvm::Log2_64_Ceil(mapped.packFactor) : 0;
  if (auto attr = loop->getAttrOfType<IntegerAttr>("bits.pass_folds"))
    mapped.passFolds = std::max<int64_t>(0, attr.getInt());
  // A reduction normally spends its first wave producing what the rest fold
  // into; an expansion that peeled that seed out folds on every wave, and says
  // so, because the wave count cannot be read both ways.
  int64_t combines = std::max<int64_t>(0, mapped.waves - 1);
  if (auto attr = loop->getAttrOfType<IntegerAttr>("bits.reduce_combines"))
    combines = std::max<int64_t>(0, attr.getInt());
  mapped.seedWaves = std::max<int64_t>(0, mapped.waves - combines);
  return mapped;
}

//===----------------------------------------------------------------------===//
// Bringing a reduction back together
//===----------------------------------------------------------------------===//

/// What a reduction spread over banks needs, read off the body.
///
/// A bank folds the tiles it was given into an accumulator of its own, so at the
/// end of the waves there is one partial result per bank, and one more per
/// packed lane within a bank. Both have to be brought together, and neither can
/// be done where they lie: a bank cannot read another bank's rows, and an
/// activation drives a whole row rather than a lane of it. So each step is a
/// copy that puts the other operand somewhere this bank's adder can reach, then
/// the fold itself.
struct Reduction {
  /// The ops the fold turned into, which the seeding wave leaves out.
  SmallVector<Operation *> ops;
  /// The accumulator's run and how many rows it has.
  int64_t accBase = 0;
  int64_t accRows = 0;
  /// The run the wave leaves its contribution in, which the seeding wave writes
  /// straight into the accumulator instead, and how many rows of it there are.
  int64_t addendBase = -1;
  int64_t addendRows = 0;
  /// Where a copy puts the partial sum being folded in.
  int64_t stageBase = 0;
  bool isAdd = false;

  bool valid() const { return !ops.empty() && accRows > 0; }
};

FailureOr<Reduction> readReduction(MappedLoop &mapped, RowBinder &binder) {
  Reduction reduction;
  for (Operation &op : mapped.loop.getBody()->without_terminator())
    if (op.hasAttr("bits.is_reduce_op"))
      reduction.ops.push_back(&op);
  if (reduction.ops.empty())
    return reduction;

  if (mapped.loop.getInits().size() != 1)
    return mapped.loop.emitError()
           << "a reduction carries exactly one accumulator; this loop carries "
           << mapped.loop.getInits().size();
  Value accumulator = mapped.loop.getRegionIterArg(0);
  auto accType = dyn_cast<SliceType>(accumulator.getType());
  const SliceLoc *accLoc = binder.sliceFor(accumulator);
  if (!accType || !accLoc || accLoc->broadcast)
    return mapped.loop.emitError()
           << "a reduction accumulates into a run of rows";
  reduction.accBase = accLoc->base;
  reduction.accRows = accType.getBitWidth();

  // The other operand of the fold is what one wave produced. Which run that is
  // is not written down anywhere: it is whichever run the fold reads that is not
  // the accumulator's.
  for (Operation *op : reduction.ops) {
    op->walk([&](ExtractRowOp extract) {
      const SliceLoc *from = binder.sliceFor(extract.getSlice());
      if (!from || from->broadcast || from->base == reduction.accBase)
        return;
      reduction.addendBase = from->base;
      if (auto slice = dyn_cast<SliceType>(extract.getSlice().getType()))
        reduction.addendRows = slice.getBitWidth();
    });
    op->walk([&](RowAddOp) { reduction.isAdd = true; });
  }
  if (reduction.addendBase < 0) {
    // A reduction whose elements each start in a bank of their own folds
    // nothing per wave -- the tree is the whole of it -- so the fold reads the
    // accumulator and whatever the tree copied in beside it, and there is no
    // third run for a wave's contribution to be in.
    if (mapped.waves > mapped.seedWaves)
      return mapped.loop.emitError()
             << "a reduction folds in something that is not in a run of rows";
    reduction.addendBase = reduction.accBase;
    reduction.addendRows = reduction.accRows;
  }

  // Where a partial sum arriving from another bank is put. A nest that folds
  // per wave needs that to be somewhere other than the run its waves write, but
  // a nest that folds only across banks has no wave writing anything: the run
  // its body names for the addend is never written by anything else, so it is
  // the staging run, and allocating a second one would leave the first dead.
  reduction.stageBase = mapped.waves > mapped.seedWaves
                            ? binder.allocateRun(reduction.accRows).base
                            : reduction.addendBase;
  return reduction;
}

/// Issues one fold of `stage` into `acc`, on whichever banks are in scope.
///
/// This is not the body's fold re-emitted. The body folds one wave's
/// contribution into the accumulator, and that contribution can be narrower
/// than the accumulator is -- a block of a BNN dot product counts to 64 and
/// accumulates into 13 bits -- so its expansion reads the rows it does not have
/// from the reserved zero row. What crosses between banks is a whole partial
/// sum, every row of it, so the fold that adds one is the accumulator's own
/// width throughout. Same command count, different rows, which is exactly why
/// it is built here rather than borrowed.
LogicalResult emitFold(OpBuilder &builder, Location loc, RowOpEmitter &emitter,
                       ScheduleEmitter &schedule, const Reduction &reduction,
                       PudBackend backend, IRMapping &map, Operation *diagOp) {
  // When the body's fold already reads every row of the accumulator, it IS this
  // fold -- the addend arrives in the staging run instead of the run a wave
  // produced, which the alias around this says, and nothing else differs. That
  // is the usual case, and it is how a fold that is not a sum -- a running
  // minimum, say -- gets across banks at all.
  if (reduction.addendRows == reduction.accRows)
    return schedule.emit(reduction.ops, builder, map);

  if (!reduction.isAdd)
    return diagOp->emitError()
           << "only a sum folds across banks so far; this reduction folds with "
              "something else, and the fold it needs has no expansion here";

  const RowOpTemplate *tmpl =
      RowOpLibrary::get().getTemplate(backend, RowOpKind::FullAdd);
  if (!tmpl)
    return diagOp->emitError() << "no adder template for backend '"
                               << getBackendName(backend) << "'";

  // The carry starts at zero and stays in the cell for the whole width, which is
  // the one command an addition costs beyond its bits.
  emitter.emitAcrossBanks(builder, loc, /*isAp=*/false,
                          RowLoc::controlRow(ControlRow::DCC1),
                          RowLoc::controlRow(ControlRow::C_0), map);
  for (int64_t row = 0; row < reduction.accRows; ++row) {
    RowLoc accRow = RowLoc::dataRow(reduction.accBase, {}, row);
    RowLoc stageRow = RowLoc::dataRow(reduction.stageBase, {}, row);
    if (failed(emitter.emitTemplate(builder, loc, *tmpl, {accRow, stageRow},
                                    {accRow}, map)))
      return diagOp->emitError() << "could not instantiate the adder a fold is";
  }
  return success();
}

/// Issues everything that happens after a bank has run all its waves.
///
/// First the packed lanes are folded within each bank, then the banks fold into
/// one another along the tree the mapping chose. Every level is the same two
/// steps -- copy, then add -- and differs only in where the copy comes from.
LogicalResult emitReduceTree(OpBuilder &builder, Location loc,
                             RowOpEmitter &emitter, ScheduleEmitter &schedule,
                             MappedLoop &mapped, const Reduction &reduction,
                             PudBackend backend, IRMapping &map) {
  auto copyAlong = [&](ArrayRef<std::pair<int64_t, int64_t>> edges) {
    emitter.emitCopies(builder, loc, edges, reduction.accBase,
                       reduction.stageBase, reduction.accRows);
  };

  // The lanes packed into one row are not addressable on their own, so folding
  // them is a copy within the bank and an addition, as many times over as it
  // takes to halve them down to one.
  SmallVector<std::pair<int64_t, int64_t>> withinBank;
  for (int64_t bank : mapped.banks)
    withinBank.push_back({bank, bank});
  for (int64_t level = 0; level < mapped.foldLevels; ++level) {
    copyAlong(withinBank);
    if (failed(emitFold(builder, loc, emitter, schedule, reduction, backend,
                        map, mapped.loop)))
      return failure();
  }

  // Then what the passes of an earlier nest left in this bank. Each of them is
  // a run further into the accumulator's rows -- that is where the pass that
  // produced it put it -- so the copy that stages one reads from a fixed offset
  // rather than from a pass this schedule is on. The last pass comes first, so
  // that what everything ends up folded into is the first, which is the run the
  // tree and whatever reads this nest go to.
  for (int64_t pass = mapped.passFolds; pass >= 1; --pass) {
    emitter.emitCopies(builder, loc, withinBank,
                       reduction.accBase + pass * reduction.accRows,
                       reduction.stageBase, reduction.accRows);
    if (failed(emitFold(builder, loc, emitter, schedule, reduction, backend,
                        map, mapped.loop)))
      return failure();
  }

  // Then across banks. Every group runs the same tree, offset to its own banks.
  int64_t groups = mapped.banks.size() / std::max<int64_t>(1, mapped.banksPerGroup);
  for (const auto &level : mapped.tree) {
    SmallVector<int64_t> destinations;
    SmallVector<std::pair<int64_t, int64_t>> edges;
    for (int64_t group = 0; group < groups; ++group) {
      int64_t offset = group * mapped.groupStride;
      for (auto [src, dst] : level) {
        edges.push_back({src + offset, dst + offset});
        destinations.push_back(dst + offset);
      }
    }
    copyAlong(edges);
    // Only the banks that were copied into have anything to add, and they all
    // do it at once.
    RowOpEmitter::BankScope scope(emitter, destinations);
    if (failed(emitFold(builder, loc, emitter, schedule, reduction, backend,
                        map, mapped.loop)))
      return failure();
  }
  return success();
}

/// Whether the nests that hand each other data were planned to the same shape.
///
/// A nest writes its result with the bank left open: one tile per bank per
/// pass, tile `t` in the `t`-th of the banks it was mapped onto. The nest that
/// reads that result addresses those rows the same way, so the two only ever
/// name the same tile if they agree on which bank holds which one, and on how
/// many each bank holds. The mapping plans them separately, so agreeing is not
/// automatic.
///
/// This cannot be left to `verifyReadsHaveWriters`. That check is about rows,
/// and a consumer that covers a fraction of the tiles, or covers them in a
/// different order, still reads nothing but rows that were written -- which is
/// exactly how the ReLU covering eight of sixty-four tiles passed it. What
/// found that was the op count; this is the check that would have.
LogicalResult verifyNestShapesAgree(MutableArrayRef<MappedLoop> loops,
                                    const RowBinder &binder,
                                    int64_t totalBanks) {
  auto normalise = [&](ArrayRef<int64_t> banks) {
    SmallVector<int64_t> out;
    int64_t total = std::max<int64_t>(1, totalBanks);
    for (int64_t bank : banks)
      out.push_back(((bank % total) + total) % total);
    return out;
  };
  // How many tiles of the operand each bank holds once the nest is done. Every
  // wave of a reduction folds into the one accumulator, so a reduction leaves
  // one per pass; anything else leaves one per wave.
  auto tilesPerBank = [&](MappedLoop &mapped) {
    bool reduces = llvm::any_of(
        mapped.loop.getBody()->without_terminator(),
        [](Operation &op) { return op.hasAttr("bits.is_reduce_op"); });
    return (reduces ? 1 : mapped.waves) * mapped.repeat;
  };

  SmallVector<llvm::DenseSet<int64_t>> reads(loops.size()), writes(loops.size());
  for (auto [i, mapped] : llvm::enumerate(loops)) {
    mapped.loop.getBody()->walk([&, i = i](InsertRowOp insert) {
      if (const SliceLoc *loc = binder.sliceFor(insert.getSlice()))
        if (!loc->broadcast)
          writes[i].insert(loc->base);
    });
    mapped.loop.getBody()->walk([&, i = i](ExtractRowOp extract) {
      if (const SliceLoc *loc = binder.sliceFor(extract.getSlice()))
        if (!loc->broadcast)
          reads[i].insert(loc->base);
    });
  }

  LogicalResult result = success();
  for (size_t consumer = 1; consumer < loops.size(); ++consumer) {
    // A fold of an earlier nest's result reads every bank that nest wrote and
    // brings them together; it is not one tile per bank and has nothing to
    // agree about. Its banks are the producer's by construction -- the tree
    // names them.
    if (loops[consumer].loop->hasAttr("bits.cross_nest_reduce"))
      continue;
    for (size_t producer = 0; producer < consumer; ++producer) {
      if (llvm::none_of(reads[consumer], [&](int64_t run) {
            return writes[producer].contains(run);
          }))
        continue;
      SmallVector<int64_t> want = normalise(loops[producer].banks);
      SmallVector<int64_t> have = normalise(loops[consumer].banks);
      if (want != have) {
        loops[consumer].loop.emitError()
            << "reads a nest that ran on other banks: this nest holds tile t "
               "in bank "
            << (have.empty() ? -1 : have[0]) << " onwards over " << have.size()
            << " banks, the nest it reads left it in bank "
            << (want.empty() ? -1 : want[0]) << " onwards over " << want.size()
            << "; every row it reads was written, but not by the tile it means";
        result = failure();
        continue;
      }
      int64_t wantTiles = tilesPerBank(loops[producer]);
      int64_t haveTiles = tilesPerBank(loops[consumer]);
      if (wantTiles != haveTiles) {
        loops[consumer].loop.emitError()
            << "reads a nest that left a different number of tiles in each "
               "bank: this nest takes "
            << haveTiles << " per bank, the nest it reads left " << wantTiles;
        result = failure();
      }
    }
  }
  return result;
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

    auto mappedLoops = readMappedLoops(func);
    if (failed(mappedLoops))
      return signalPassFailure();

    // Every bank running a computation holds its operands in the same rows, so
    // the layout is decided once and the bank left open. One allocator for the
    // whole function: the nests run one after another but their operands are all
    // in the machine at once.
    int64_t layoutBank = mappedLoops->front().banks.empty()
                             ? 0
                             : mappedLoops->front().banks.front();
    pud::AddressAllocator allocator(*geometry);
    RowBinder binder(allocator, layoutBank);
    for (MappedLoop &mapped : *mappedLoops) {
      // Waves that fold are what makes the body read anything. Only a reduction
      // has a seeding wave to subtract, so only a reduction can have none left.
      bool reduces = llvm::any_of(
          mapped.loop.getBody()->without_terminator(),
          [](Operation &op) { return op.hasAttr("bits.is_reduce_op"); });
      binder.beginNest(mapped.loop, mapped.waves, mapped.repeat,
                       !reduces || mapped.waves > mapped.seedWaves,
                       operandPerWave);
      if (failed(binder.bindCarried(mapped.loop)) ||
          failed(binder.bind(*mapped.loop.getBody())))
        return signalPassFailure();
    }

    // A reduction that folds only across banks suppresses its body's fold on
    // the one wave it runs, so those ops issue nothing and read nothing; the
    // fold that does happen is the tree's, on the accumulator and the run the
    // tree copies into.
    llvm::DenseSet<Operation *> unemitted;
    for (MappedLoop &mapped : *mappedLoops) {
      if (mapped.waves > mapped.seedWaves)
        continue;
      for (Operation &op : mapped.loop.getBody()->without_terminator())
        if (op.hasAttr("bits.is_reduce_op"))
          unemitted.insert(&op);
    }

    // Everything is bound and nothing has been emitted, which is the only
    // moment at which both where a value lives and where it came from are known.
    if (failed(verifyReadsHaveWriters(func, binder, unemitted)))
      return signalPassFailure();
    if (failed(verifyNestShapesAgree(*mappedLoops, binder,
                                     geometry->totalBanks)))
      return signalPassFailure();

    // The new body is built beside the old one and swapped in at the end. It is
    // built from the old one -- the loop nest, the index arithmetic, the order
    // -- so the old one has to still be there while it is being read.
    Location loc = func.getLoc();
    auto entry = std::make_unique<Block>();
    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(entry.get());

    for (MappedLoop &mapped : *mappedLoops) {
      RowOpEmitter emitter(*geometry, binder, mapped.banks);
      ScheduleEmitter schedule(emitter, binder, target);

      IRMapping map;
      // Row indices are built from constants defined outside the mapped loop, so
      // bring those across before anything asks for them -- from every block the
      // loop is nested in, outermost first, since a constant may sit anywhere
      // above it.
      SmallVector<Block *> enclosing;
      for (Operation *parent = mapped.loop; parent;
           parent = parent->getParentOp())
        if (Block *block = parent->getBlock())
          enclosing.push_back(block);
      for (Block *block : llvm::reverse(enclosing))
        for (Operation &op : *block) {
          if (!isa<arith::ArithDialect>(op.getDialect()))
            continue;
          // Only what can stand on its own out here. An enclosing loop's index
          // arithmetic is about which tile that loop is on, and this schedule
          // has no such iteration to be on -- bringing it across would leave the
          // new body pointing at the old one.
          if (llvm::any_of(op.getOperands(),
                           [&](Value v) { return !map.contains(v); }))
            continue;
          builder.clone(op, map);
        }

      auto reduction = readReduction(mapped, binder);
      if (failed(reduction))
        return signalPassFailure();

      Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
      Value repeats = builder.create<arith::ConstantIndexOp>(loc, mapped.repeat);
      int64_t seedWaves = reduction->valid() ? mapped.seedWaves : 0;
      Value seeds = builder.create<arith::ConstantIndexOp>(loc, seedWaves);
      Value tiles =
          builder.create<arith::ConstantIndexOp>(loc, mapped.waves - seedWaves);

      // Enclosing loops whose iterations did not all fit in the machine at once
      // run the whole schedule again for the ones that were left.
      auto repeatLoop = builder.create<scf::ForOp>(loc, zero, repeats, one);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(repeatLoop.getBody());

        // Where in the operand this pass is. A run that holds one pass's values
        // is addressed this many runs into the space it was given, so these are
        // what stop the pass after this one writing over what it leaves.
        auto asI64 = [&](Value index) {
          return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    index)
              .getResult();
        };
        Value pass = asI64(repeatLoop.getInductionVar());
        Value firstWave = builder.create<arith::MulIOp>(
            loc, pass,
            builder.create<arith::ConstantIntOp>(loc, mapped.waves, 64));

        // What the tree can refer to: everything already in scope out here. The
        // waves clone their index arithmetic inside the loops that run them, and
        // the tree comes after those loops, so it cannot use any of it.
        IRMapping outerScope = map;

        // A reduction has nothing to fold into until something has been
        // produced, so its first wave produces and stops. Writing that first
        // contribution where the accumulator lives is what makes it the
        // accumulator's initial value, and it costs nothing: the same commands,
        // addressed one run over.
        if (seedWaves > 0) {
          auto seedLoop = builder.create<scf::ForOp>(loc, zero, seeds, one);
          OpBuilder::InsertionGuard seedGuard(builder);
          builder.setInsertionPointToStart(seedLoop.getBody());
          emitter.setPass(builder.create<arith::AddIOp>(
                              loc, firstWave,
                              asI64(seedLoop.getInductionVar())),
                          pass);
          emitter.aliasRun(reduction->addendBase, reduction->accBase);
          schedule.suppress(reduction->ops);
          // On its own map: what this wave brings across lands inside its loop,
          // where the waves that follow it cannot see it.
          IRMapping seedScope = map;
          if (failed(schedule.emit(*mapped.loop.getBody(), builder, seedScope)))
            return signalPassFailure();
          schedule.emitEverything();
          emitter.clearAliases();
        }

        // The mapped loop ran once per tile of the whole operand; each bank
        // takes its own share of them, the tiles sharing a bank row run as one
        // wave, and the banks run at the same time.
        auto tileLoop = builder.create<scf::ForOp>(loc, zero, tiles, one);
        {
          OpBuilder::InsertionGuard tileGuard(builder);
          builder.setInsertionPointToStart(tileLoop.getBody());
          // The waves the seed already ran come before this one.
          Value wave = builder.create<arith::AddIOp>(
              loc, firstWave, asI64(tileLoop.getInductionVar()));
          if (seedWaves > 0)
            wave = builder.create<arith::AddIOp>(
                loc, wave,
                builder.create<arith::ConstantIntOp>(loc, seedWaves, 64));
          emitter.setPass(wave, pass);
          if (failed(schedule.emit(*mapped.loop.getBody(), builder, map)))
            return signalPassFailure();
        }

        // Now every bank holds a partial result, and they have to become one.
        if (reduction->valid() &&
            (mapped.foldLevels > 0 || mapped.passFolds > 0 ||
             !mapped.tree.empty())) {
          // The tree runs once the waves are done, so there is no wave to be
          // on. What it touches is the accumulator, which steps with the pass,
          // and the staging run, which is scratch; a run that steps with the
          // wave has no address out here, and the emitter says so if one is
          // asked for.
          emitter.setPass(Value(), pass);
          emitter.aliasRun(reduction->addendBase, reduction->stageBase);
          if (failed(emitReduceTree(builder, loc, emitter, schedule, mapped,
                                    *reduction, target, outerScope)))
            return signalPassFailure();
          if (emitter.strayPassRow) {
            mapped.loop.emitError()
                << "the reduce tree addresses a run that holds one wave's "
                   "values; the tree runs after the waves and has no wave to "
                   "take that run's rows from";
            return signalPassFailure();
          }
          emitter.clearAliases();
        }
      }
    }

    builder.setInsertionPointToEnd(entry.get());
    builder.create<func::ReturnOp>(loc);

    Region &body = func.getBody();
    body.getBlocks().clear();
    body.push_back(entry.release());
    func.setType(FunctionType::get(func.getContext(), {}, {}));

    // A row op activates a whole subarray, so every row it names has to be in
    // one -- and the rows the substrate reserves for computation are at the
    // front of the space, so every data row has to be in theirs. That makes the
    // subarray the size of the row space, not a boundary it may cross at a
    // cost. Saying so here turns what would otherwise be an assertion failing
    // deep inside a simulated run into a number that can be compared against
    // the machine.
    int64_t dataRows = allocator.rowsAllocated(layoutBank);
    int64_t reserved = pud::controlRowCount();
    int64_t subarrayRows = rowsPerSubarray;
    if (subarrayRows > 0 && reserved + dataRows > subarrayRows) {
      func.emitError()
          << "the kernel addresses " << dataRows << " rows a bank, which with "
          << reserved << " reserved for computation is " << (reserved + dataRows)
          << " in a subarray of " << subarrayRows
          << "; a row op cannot name rows in two subarrays, so every row of "
             "this kernel has to be in one. More banks is what makes an "
             "operand fit; a kernel too big to be resident at any bank count "
             "can still be measured with operand-per-wave=false, which shares "
             "a pass's rows between its waves and so computes on the same tile "
             "several times over";
      return signalPassFailure();
    }

    // The shape of the row space these addresses were handed out in. Whatever
    // realises them has to provide exactly this many rows per bank, and has to
    // split a flat bank index into a channel and a bank on the same boundary,
    // so the shape travels with the IR rather than being restated downstream.
    if (auto module = func->getParentOfType<ModuleOp>()) {
      module->setAttr("pud.data_rows_per_bank",
                      builder.getI64IntegerAttr(dataRows));
      module->setAttr("pud.banks_per_channel",
                      builder.getI64IntegerAttr(geometry->banksPerChannel));
      module->setAttr("pud.total_banks",
                      builder.getI64IntegerAttr(geometry->totalBanks));
      // The stride the machine decodes rows at, which is the machine's to say
      // and not this pass's -- the option is an override for asking what some
      // other row size would cost.
      int64_t stride =
          rowStride > 0 ? (int64_t)rowStride : geometry->rowStride;
      module->setAttr("pud.row_stride", builder.getI64IntegerAttr(stride));
      // What one channel's controller owns, which is what separates a bank in
      // one channel from the same-numbered bank in the next. It is the span a
      // subarray's rows take across a channel's banks, rounded up to a power of
      // two -- the same figure both gem5 configurations compute, because the
      // controller's geometry checks want it that way.
      int64_t channelRows = std::max<int64_t>(1, subarrayRows);
      int64_t channelSpan = channelRows * geometry->banksPerChannel * stride;
      int64_t channelSize = 1;
      while (channelSize < channelSpan)
        channelSize <<= 1;
      module->setAttr("pud.channel_size",
                      builder.getI64IntegerAttr(channelSize));
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::bits::createConvertBitsRowsToPuDPass() {
  return std::make_unique<ConvertBitsRowsToPuD>();
}
