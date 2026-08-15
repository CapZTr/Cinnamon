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
// value semantics to preserve: it names rows and leaves its result in them, so
// once operands are bound to addresses the SSA values that stood for rows have
// nothing left to say.
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
// Data rows
//===----------------------------------------------------------------------===//

/// Where a computation's operands live, per bank.
///
/// A bit-serial op reads bit j of an operand from the j-th row of that operand's
/// range, so each operand is a run of consecutive rows as wide as it is.
struct DataRows {
  pud::RowAddress lhs;
  pud::RowAddress rhs;
  pud::RowAddress out;
  pud::RowAddress temp;

  /// Reserves the rows one bank needs. Every bank running this computation gets
  /// the same rows, so the allocator is asked once per bank.
  static DataRows allocate(pud::AddressAllocator &allocator, int64_t flatBank,
                           int64_t width, int64_t numTemps) {
    DataRows rows;
    rows.lhs = allocator.allocate(flatBank, width);
    rows.rhs = allocator.allocate(flatBank, width);
    rows.out = allocator.allocate(flatBank, width);
    rows.temp = allocator.allocate(flatBank, numTemps);
    return rows;
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
  /// State the row op carries across bit iterations in a dual-contact cell, and
  /// therefore has to seed once per tile -- for an add, a carry starting at
  /// zero. `carryRow` is the cell the template keeps it in.
  bool seedsCarry = false;
  ControlRow carryRow = ControlRow::DCC1;
  ControlRow carrySeed = ControlRow::C_0;
};

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Builds the addresses a template's operands name, and issues its commands.
class RowOpEmitter {
public:
  /// A row within a bank, with the bank left open: which bank fills in per
  /// iteration of the loop that issues the command.
  struct RowRef {
    bool valid = false;
    bool isControl = false;
    /// For a control row, which one.
    ControlRow control = ControlRow::T0;
    /// For a data row, the first row of the operand's range and the offset into
    /// it -- the bit being processed.
    int64_t baseRow = 0;
    Value offset;

    static RowRef controlRow(ControlRow row) {
      RowRef ref;
      ref.valid = true;
      ref.isControl = true;
      ref.control = row;
      return ref;
    }
    static RowRef dataRow(const pud::RowAddress &addr, Value offset) {
      RowRef ref;
      ref.valid = true;
      ref.baseRow = addr.row;
      ref.offset = offset;
      return ref;
    }
  };

  RowOpEmitter(OpBuilder &builder, const pud::MemoryGeometry &geometry,
               const DataRows &rows)
      : builder(builder), geometry(geometry), rows(rows) {}

  /// Issues one command to every bank in `[bankBase, bankBase + bankCount)`.
  ///
  /// The banks are the inner loop. They are independent of one another, so
  /// getting a command to all of them before starting the next is what turns
  /// that independence into overlapped activations.
  void emitAcrossBanks(Location loc, bool isAp, const RowRef &dst,
                       const RowRef &src, int64_t bankBase, int64_t bankCount) {
    Value lb = builder.create<arith::ConstantIndexOp>(loc, bankBase);
    Value ub = builder.create<arith::ConstantIndexOp>(loc, bankBase + bankCount);
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
    builder.create<scf::ForOp>(
        loc, lb, ub, step, ValueRange{},
        [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
          BankCoord coord = decomposeBank(b, bodyLoc, iv);
          Value dstAddr = materialise(b, bodyLoc, dst, coord);
          if (isAp) {
            b.create<pud::APOp>(bodyLoc, dstAddr, pud::Backend::Simdram);
          } else {
            Value srcAddr = materialise(b, bodyLoc, src, coord);
            b.create<pud::AAPOp>(bodyLoc, srcAddr, dstAddr,
                                 pud::Backend::Simdram);
          }
          b.create<scf::YieldOp>(bodyLoc);
        });
  }

  /// Instantiates `tmpl` for one bit position: the template's i-th data operand
  /// reads the bit-th row of that operand's range.
  LogicalResult emitTemplate(Location loc, const RowOpTemplate &tmpl,
                             Value bitIv, const RowSchedule &sched) {
    SmallVector<RowRef, 3> inputs{RowRef::dataRow(rows.lhs, bitIv),
                                  RowRef::dataRow(rows.rhs, bitIv)};
    SmallVector<RowRef, 2> outputs{RowRef::dataRow(rows.out, bitIv)};

    for (const RowOpInstr &instr : tmpl.instrs) {
      RowRef dst = resolve(loc, instr.dst, inputs, outputs);
      RowRef src = instr.isAp ? RowRef{} : resolve(loc, instr.src, inputs, outputs);
      if (!dst.valid || (!instr.isAp && !src.valid))
        return failure();
      emitAcrossBanks(loc, instr.isAp, dst, src, sched.bankBase,
                      sched.bankCount);
    }
    return success();
  }

private:
  /// A bank, in the terms the memory is organised in.
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

  Value materialise(OpBuilder &b, Location loc, const RowRef &ref,
                    const BankCoord &coord) {
    if (ref.isControl)
      return b.create<pud::GetCtrlRowOp>(
          loc, pud::RowType::get(b.getContext(), groupOf(ref.control)),
          coord.channel, coord.rank, coord.bank, coord.subarray,
          toDialect(ref.control));

    Value base = b.create<arith::ConstantIntOp>(loc, ref.baseRow, 64);
    Value row = ref.offset
                    ? b.create<arith::AddIOp>(loc, base, ref.offset).getResult()
                    : base;
    return b.create<pud::GetRowOp>(
        loc, pud::RowType::get(b.getContext(), kDataGroup), coord.channel,
        coord.rank, coord.bank, coord.subarray, row);
  }

  RowRef resolve(Location loc, const RowOperand &operand,
                 ArrayRef<RowRef> inputs, ArrayRef<RowRef> outputs) {
    switch (operand.kind) {
    case RowOperand::Kind::Control:
      return RowRef::controlRow(operand.getControlRow());
    case RowOperand::Kind::Input:
      return operand.index < (int)inputs.size() ? inputs[operand.index]
                                                : RowRef{};
    case RowOperand::Kind::Output:
      return operand.index < (int)outputs.size() ? outputs[operand.index]
                                                 : RowRef{};
    case RowOperand::Kind::Temp: {
      Value offset =
          builder.create<arith::ConstantIntOp>(loc, operand.index, 64);
      return RowRef::dataRow(rows.temp, offset);
    }
    }
    return RowRef{};
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

  OpBuilder &builder;
  pud::MemoryGeometry geometry;
  DataRows rows;
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

/// Whether an op computes on rows, whatever this pass can currently do with it.
bool isRowLevelOp(Operation *op) {
  return isa<RowAddOp, RowAndOp, RowOrOp, RowXOrOp, MajOp, NotOp>(op);
}

/// Slice-level ops `bits-pattern-apply` has no row expansion for yet. They are
/// what stands between a kernel and this pass, so they are worth naming.
bool isUnexpandedSliceOp(Operation *op) {
  return isa<AddIOp, AddIFullOp, SubIOp, MulIOp, AndOp, OrOp, XOrOp, XNOrOp,
             RangeScanOp, MaxOp, MinOp, MuxOp>(op);
}

FailureOr<RowSchedule> readSchedule(func::FuncOp func) {
  SmallVector<Operation *> rowOps;
  SmallVector<Operation *> sliceOps;
  func.walk([&](Operation *op) {
    if (isRowLevelOp(op))
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
  if (rowOps.size() != 1 || !isa<RowAddOp>(rowOps.front())) {
    llvm::SmallDenseSet<StringRef> names;
    for (Operation *op : rowOps)
      names.insert(op->getName().getStringRef());
    auto diag = rowOps.front()->emitError()
                << "this pass lowers a single bits.row_add per function; this "
                   "one has " << rowOps.size() << " row ops:";
    for (StringRef name : names)
      diag << " " << name;
    diag.attachNote()
        << "lowering more needs each row-level value bound to the row it "
           "lives in, rather than a schedule read off one op";
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

    auto geometry = readGeometry(func);
    if (failed(geometry))
      return signalPassFailure();

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

    // Every bank running this computation holds its operands in the same rows,
    // so the layout is decided once and the bank left open.
    pud::AddressAllocator allocator(*geometry);
    DataRows rows = DataRows::allocate(allocator, sched->bankBase, sched->width,
                                       /*numTemps=*/4);

    // Replace the body wholesale: the tensor dataflow described which rows to
    // touch, and the addresses now say it directly.
    Location loc = func.getLoc();
    Region &body = func.getBody();
    body.getBlocks().clear();
    Block *entry = new Block();
    body.push_back(entry);
    func.setType(FunctionType::get(func.getContext(), {}, {}));

    OpBuilder builder(entry, entry->begin());

    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value tiles =
        builder.create<arith::ConstantIndexOp>(loc, sched->itersPerBank);
    Value bits = builder.create<arith::ConstantIndexOp>(loc, sched->width);

    auto tileLoop = builder.create<scf::ForOp>(loc, zero, tiles, one);
    builder.setInsertionPointToStart(tileLoop.getBody());
    {
      RowOpEmitter emitter(builder, *geometry, rows);
      if (sched->seedsCarry) {
        // One command per tile seeds the carry; the template keeps it resident
        // in the cell for the whole bit loop, which is what makes an n-bit add
        // 8n + 1 commands rather than 9n.
        emitter.emitAcrossBanks(
            loc, /*isAp=*/false,
            RowOpEmitter::RowRef::controlRow(sched->carryRow),
            RowOpEmitter::RowRef::controlRow(sched->carrySeed),
            sched->bankBase, sched->bankCount);
      }

      auto bitLoop = builder.create<scf::ForOp>(loc, zero, bits, one);
      builder.setInsertionPointToStart(bitLoop.getBody());
      RowOpEmitter bitEmitter(builder, *geometry, rows);
      Value bitIv = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), bitLoop.getInductionVar());
      if (failed(bitEmitter.emitTemplate(loc, *tmpl, bitIv, *sched)))
        return signalPassFailure();
    }

    builder.setInsertionPointToEnd(entry);
    builder.create<func::ReturnOp>(loc);

    // The shape of the row space these addresses were handed out in. Whatever
    // realises them has to provide exactly this many rows per bank, and has to
    // split a flat bank index into a channel and a bank on the same boundary,
    // so the shape travels with the IR rather than being restated downstream.
    if (auto module = func->getParentOfType<ModuleOp>()) {
      module->setAttr("pud.data_rows_per_bank",
                      builder.getI64IntegerAttr(
                          allocator.rowsAllocated(sched->bankBase)));
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
