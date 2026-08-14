//===- RowOpLibrary.cpp - row-op primitives compiled by lime --------------===//

#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"

#include "eggmock.hpp"
#include "gp.h"
#include "mockturtle/networks/mig.hpp"
#include "utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstring>
#include <string>

using namespace mlir::bits;
using MIG = mockturtle::mig_network;

namespace {

/// Index of the dual-contact cell type in lime's cell type list.
///
/// The generated cell-type enum is built from a BTreeMap keyed by name, so the
/// ordering is the constant type followed by the cell type names sorted
/// alphabetically -- for SIMDRAM: bool, D, DCC, T. Hints have to name this index
/// before a program exists to read the names from, so it is fixed here and
/// checked against the exported names afterwards.
constexpr uint32_t kDccCellTypeIdx = 2;

//===----------------------------------------------------------------------===//
// MIG builders
//===----------------------------------------------------------------------===//

MIG buildAnd() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi();
  n.create_po(n.create_maj(a, b, n.get_constant(false)));
  return n;
}

MIG buildOr() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi();
  n.create_po(n.create_maj(a, b, n.get_constant(true)));
  return n;
}

MIG buildNot() {
  MIG n;
  auto a = n.create_pi();
  n.create_po(!a);
  return n;
}

MIG buildMaj3() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi(), c = n.create_pi();
  n.create_po(n.create_maj(a, b, c));
  return n;
}

/// XNOR as (a | !b) & (!a | b). Three majorities either way; the dual-contact
/// cells hand back both complements without a command of their own.
MIG buildXnor() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi();
  auto t = n.get_constant(true), f = n.get_constant(false);
  n.create_po(n.create_maj(n.create_maj(a, !b, t), n.create_maj(!a, b, t), f));
  return n;
}

MIG buildXor() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi();
  auto t = n.get_constant(true), f = n.get_constant(false);
  n.create_po(n.create_maj(n.create_maj(a, b, t), !n.create_maj(a, b, f), f));
  return n;
}

/// Full adder. Input 2 is the carry-in and output 1 the carry-out, which is what
/// the loop-carry hints below pin into a dual-contact cell.
MIG buildFullAdd() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi(), c = n.create_pi();
  auto cout = n.create_maj(a, b, c);
  auto m = n.create_maj(a, !b, c);
  n.create_po(n.create_maj(b, !cout, m)); // sum
  n.create_po(cout);
  return n;
}

/// Full subtractor: the adder with the subtrahend complemented. In a MIG the
/// complement is a free edge attribute, but it still has to be materialised in a
/// row because the adder reads the subtrahend twice per bit.
MIG buildFullSub() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi(), c = n.create_pi();
  auto cout = n.create_maj(a, !b, c);
  auto m = n.create_maj(a, b, c);
  n.create_po(n.create_maj(!b, !cout, m)); // difference
  n.create_po(cout);
  return n;
}

/// ReLU bit step: x & !sign.
MIG buildRelu() {
  MIG n;
  auto x = n.create_pi(), sign = n.create_pi();
  n.create_po(n.create_maj(x, !sign, n.get_constant(false)));
  return n;
}

/// Compare-and-select bit step, emitting both min and max given an "a greater"
/// flag.
MIG buildMinMax() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi(), g = n.create_pi();
  auto f = n.get_constant(false), t = n.get_constant(true);
  auto ite = [&](auto s, auto x, auto y) {
    return n.create_maj(n.create_maj(s, x, f), n.create_maj(!s, y, f), t);
  };
  n.create_po(ite(g, b, a)); // min
  n.create_po(ite(g, a, b)); // max
  return n;
}

/// One BitWeaving range-scan bit step:
///   R' = maj(R, v,  lower[j])     ">= lower" chain
///   S' = maj(S, !v, upper[j])     "<= upper" chain
///
/// The scan bounds are compile-time constants, so each (lower[j], upper[j]) pair
/// is its own template -- modelling them as runtime inputs would make lime load
/// them from data rows, which no correct expansion needs to do. Both
/// accumulators are loop-carried into dual-contact cells.
MIG buildRangeScan(bool lowerBit, bool upperBit) {
  MIG n;
  auto v = n.create_pi(); // 0
  auto r = n.create_pi(); // 1: R accumulator
  auto s = n.create_pi(); // 2: S accumulator
  n.create_po(n.create_maj(r, v, n.get_constant(lowerBit)));
  n.create_po(n.create_maj(s, !v, n.get_constant(upperBit)));
  return n;
}

MIG buildMig(RowOpKind kind) {
  switch (kind) {
  case RowOpKind::And:
    return buildAnd();
  case RowOpKind::Or:
    return buildOr();
  case RowOpKind::Not:
    return buildNot();
  case RowOpKind::Maj3:
    return buildMaj3();
  case RowOpKind::Xor:
    return buildXor();
  case RowOpKind::Xnor:
    return buildXnor();
  case RowOpKind::FullAdd:
  case RowOpKind::FullAddLoaded:
    return buildFullAdd();
  case RowOpKind::FullSub:
    return buildFullSub();
  case RowOpKind::Relu:
    return buildRelu();
  case RowOpKind::MinMax:
    return buildMinMax();
  case RowOpKind::RangeScan00:
    return buildRangeScan(false, false);
  case RowOpKind::RangeScan01:
    return buildRangeScan(false, true);
  case RowOpKind::RangeScan10:
    return buildRangeScan(true, false);
  case RowOpKind::RangeScan11:
    return buildRangeScan(true, true);
  case RowOpKind::NUM_KINDS:
    break;
  }
  llvm_unreachable("unhandled row-op kind");
}

//===----------------------------------------------------------------------===//
// Hints
//===----------------------------------------------------------------------===//

/// Which values stay resident in a dual-contact cell across bit iterations.
/// Getting this wrong does not produce a wrong program, only a slower one, so
/// the expected command counts below are asserted rather than trusted.
struct LoopCarry {
  llvm::SmallVector<input_hint, 2> inputs;
  llvm::SmallVector<output_hint, 2> outputs;
};

LoopCarry getLoopCarry(RowOpKind kind) {
  LoopCarry lc;
  switch (kind) {
  case RowOpKind::FullAdd:
  case RowOpKind::FullSub:
    // carry-in (input 2) and carry-out (output 1) both live in DCC[1]
    lc.inputs.push_back({2, kDccCellTypeIdx, 1});
    lc.outputs.push_back({1, kDccCellTypeIdx, 1});
    break;
  case RowOpKind::RangeScan00:
  case RowOpKind::RangeScan01:
  case RowOpKind::RangeScan10:
  case RowOpKind::RangeScan11:
    // both comparison chains stay resident, one per dual-contact cell
    lc.inputs.push_back({1, kDccCellTypeIdx, 0});
    lc.inputs.push_back({2, kDccCellTypeIdx, 1});
    lc.outputs.push_back({0, kDccCellTypeIdx, 0});
    lc.outputs.push_back({1, kDccCellTypeIdx, 1});
    break;
  default:
    // FullAddLoaded deliberately takes no hints: its carry comes from a data
    // row, which is exactly the difference from FullAdd.
    break;
  }
  return lc;
}

/// Command counts a correctly hinted template must hit. A miss means a hint did
/// not take effect, which would silently cost commands on every bit of every
/// operand -- worth failing loudly for.
std::optional<unsigned> getRequiredCommands(RowOpKind kind) {
  switch (kind) {
  case RowOpKind::FullAdd:
    return 8;
  case RowOpKind::FullAddLoaded:
    return 9;
  default:
    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// lime cell -> row slot
//===----------------------------------------------------------------------===//

/// A row set one activation drives, as the cells lime names it by.
struct GroupPattern {
  ControlRow row;
  llvm::SmallVector<std::pair<const char *, uint32_t>, 3> cells;
  llvm::SmallVector<bool, 3> inverted;
};

/// The row sets SIMDRAM wires together: two dual-row pairs involving a
/// dual-contact cell's complement, two more between compute rows, and four
/// triples whose activation leaves the majority of the three in all of them.
const std::array<GroupPattern, 8> &getSimdramGroups() {
  static const std::array<GroupPattern, 8> groups = {{
      {ControlRow::DCC0N_T0, {{"DCC", 0}, {"T", 0}}, {true, false}},
      {ControlRow::DCC1N_T1, {{"DCC", 1}, {"T", 1}}, {true, false}},
      {ControlRow::T2_T3, {{"T", 2}, {"T", 3}}, {false, false}},
      {ControlRow::T0_T3, {{"T", 0}, {"T", 3}}, {false, false}},
      {ControlRow::T0_T1_T2, {{"T", 0}, {"T", 1}, {"T", 2}}, {false, false, false}},
      {ControlRow::T0_T1_T3, {{"T", 0}, {"T", 1}, {"T", 3}}, {false, false, false}},
      {ControlRow::DCC0_T1_T3, {{"DCC", 0}, {"T", 1}, {"T", 3}}, {false, false, false}},
      {ControlRow::DCC1_T0_T2, {{"DCC", 1}, {"T", 0}, {"T", 2}}, {false, false, false}},
  }};
  return groups;
}

/// Binds lime cells to row slots and data-row placeholders for one program.
class CellResolver {
public:
  CellResolver(const ffi_program &prog) : prog(prog) {
    if (prog.num_cell_type_names <= kDccCellTypeIdx ||
        std::strcmp(prog.cell_type_names[kDccCellTypeIdx], "DCC") != 0)
      llvm::report_fatal_error(
          "lime cell type ordering changed: index 2 is no longer the "
          "dual-contact cell, so the loop-carry hints name the wrong cell type");

    for (uint64_t i = 0; i < prog.num_input_cells; i++)
      if (isData(prog.input_cells[i]))
        dataRows[prog.input_cells[i].cell_index] =
            RowOperand::input(static_cast<int>(i));
    for (uint64_t i = 0; i < prog.num_output_cells; i++)
      if (isData(prog.output_cells[i]))
        dataRows[prog.output_cells[i].cell_index] =
            RowOperand::output(static_cast<int>(i));
  }

  llvm::StringRef typeName(const ffi_cell &cell) const {
    return prog.cell_type_names[cell.cell_type_idx];
  }
  bool isData(const ffi_cell &cell) const { return typeName(cell) == "D"; }

  /// A single cell read or written on its own.
  RowOperand resolve(const ffi_cell &cell, bool inverted) {
    llvm::StringRef type = typeName(cell);
    if (type == "bool")
      return RowOperand::control(cell.cell_index != 0 ? ControlRow::C_1
                                                      : ControlRow::C_0);
    if (type == "T")
      return RowOperand::control(static_cast<ControlRow>(
          static_cast<int>(ControlRow::T0) + cell.cell_index));
    if (type == "DCC") {
      // a dual-contact cell presents its value on one row and its complement on
      // another; the two are adjacent in the enumeration
      int base = static_cast<int>(ControlRow::DCC0) + 2 * cell.cell_index;
      return RowOperand::control(
          static_cast<ControlRow>(inverted ? base + 1 : base));
    }
    if (type == "D") {
      auto it = dataRows.find(cell.cell_index);
      if (it != dataRows.end())
        return it->second;
      // a data row lime used as scratch
      auto [inserted, _] = dataRows.try_emplace(
          cell.cell_index, RowOperand::temp(nextTemp));
      nextTemp++;
      return inserted->second;
    }
    llvm::report_fatal_error("lime program uses an unknown cell type");
  }

  /// A set of cells driven by one activation. Falls back to the single-cell
  /// mapping when there is only one.
  RowOperand resolveGroup(llvm::ArrayRef<ffi_operand> operands) {
    if (operands.size() == 1)
      return resolve(operands[0].cell, operands[0].inverted != 0);
    for (const GroupPattern &group : getSimdramGroups()) {
      if (group.cells.size() != operands.size())
        continue;
      if (matches(group, operands))
        return RowOperand::control(group.row);
    }
    llvm::report_fatal_error(
        "lime emitted a row group that the SIMDRAM slot table does not name");
  }

  /// A set of cells activated together as an instruction's inputs (a TRA).
  RowOperand resolveInputGroup(llvm::ArrayRef<ffi_cell> cells) {
    if (cells.size() == 1)
      return resolve(cells[0], false);
    llvm::SmallVector<ffi_operand, 3> operands;
    for (const ffi_cell &cell : cells)
      operands.push_back({cell, 0});
    return resolveGroup(operands);
  }

private:
  bool matches(const GroupPattern &group,
               llvm::ArrayRef<ffi_operand> operands) const {
    // a group is a set: the same rows in any order name the same slot
    llvm::SmallVector<bool, 3> used(operands.size(), false);
    for (size_t i = 0; i < group.cells.size(); i++) {
      bool found = false;
      for (size_t j = 0; j < operands.size() && !found; j++) {
        if (used[j])
          continue;
        const ffi_operand &op = operands[j];
        if (typeName(op.cell) == group.cells[i].first &&
            op.cell.cell_index == group.cells[i].second &&
            (op.inverted != 0) == group.inverted[i]) {
          used[j] = true;
          found = true;
        }
      }
      if (!found)
        return false;
    }
    return true;
  }

  const ffi_program &prog;
  llvm::DenseMap<uint32_t, RowOperand> dataRows;
  int nextTemp = 0;
};

//===----------------------------------------------------------------------===//
// Compilation
//===----------------------------------------------------------------------===//

bool compileTemplate(RowOpKind kind, RowOpTemplate &out) {
  MIG mig = buildMig(kind);
  preoptimize(mig);

  LoopCarry lc = getLoopCarry(kind);
  program_holder prog;
  compiler_settings settings{};
  settings.rewriting = rewriting_strategy::none;
  settings.validator = new_validator(mig);
  settings.mode = compilation_mode::exhaustive;
  settings.candidate_selection = candidate_selection_mode::all;
  settings.hints = lc.inputs.empty() ? nullptr : lc.inputs.data();
  settings.num_hints = lc.inputs.size();
  settings.output_hints = lc.outputs.empty() ? nullptr : lc.outputs.data();
  settings.num_output_hints = lc.outputs.size();
  settings.program_out = prog.out();

  compiler_statistics stat =
      eggmock::send_ntk(mig, eggmock::receiver(gp_compile_simdram(settings)));

  out.validated = stat.validation_success;
  if (!out.validated) {
    llvm::errs() << "row-op template '" << getRowOpName(kind)
                 << "': lime could not verify the compiled program\n";
    return false;
  }

  CellResolver resolver(prog.get());
  for (const ffi_instruction &instr : prog) {
    RowOpInstr emitted;
    llvm::StringRef opcode = instr.opcode_name;
    if (opcode == "TRA") {
      // a TRA with no destination leaves the majority in the activated rows and
      // costs a single activate-precharge
      emitted.isAp = instr.num_outputs == 0;
      RowOperand activated = resolver.resolveInputGroup(
          {instr.inputs, static_cast<size_t>(instr.num_inputs)});
      if (emitted.isAp) {
        emitted.dst = activated;
      } else {
        emitted.src = activated;
        emitted.dst = resolver.resolveGroup(
            {instr.outputs, static_cast<size_t>(instr.num_outputs)});
      }
    } else {
      // RC / RC_INV: copy one row into a destination group. RC_INV reads a
      // dual-contact cell through its complementing row.
      bool inverted = opcode == "RC_INV";
      emitted.isAp = false;
      emitted.src = resolver.resolve(instr.inputs[0], inverted);
      emitted.dst = resolver.resolveGroup(
          {instr.outputs, static_cast<size_t>(instr.num_outputs)});
    }
    (emitted.isAp ? out.numAP : out.numAAP)++;
    out.instrs.push_back(emitted);
  }

  out.numInputs = static_cast<unsigned>(prog.num_input_cells());
  out.numOutputs = static_cast<unsigned>(prog.num_output_cells());
  for (uint64_t i = 0; i < prog.num_input_cells(); i++)
    out.inputBindings.push_back(resolver.resolve(prog.input_cells()[i], false));
  for (uint64_t i = 0; i < prog.num_output_cells(); i++)
    out.outputBindings.push_back(
        resolver.resolve(prog.output_cells()[i], false));

  if (auto required = getRequiredCommands(kind)) {
    if (out.totalCommands() != *required) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "row-op template '" << getRowOpName(kind) << "' compiled to "
         << out.totalCommands() << " commands but must be " << *required
         << "; the loop-carry hint did not take effect, which would cost a "
            "command on every bit of every operand";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
  }
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// Names
//===----------------------------------------------------------------------===//

llvm::StringRef mlir::bits::getBackendName(PudBackend backend) {
  switch (backend) {
  case PudBackend::Simdram:
    return "simdram";
  case PudBackend::Prada:
    return "prada";
  case PudBackend::Fcdram:
    return "fcdram";
  }
  llvm_unreachable("unhandled backend");
}

llvm::StringRef mlir::bits::getControlRowName(ControlRow row) {
  switch (row) {
  case ControlRow::T0: return "T0";
  case ControlRow::T1: return "T1";
  case ControlRow::T2: return "T2";
  case ControlRow::T3: return "T3";
  case ControlRow::DCC0: return "DCC0";
  case ControlRow::DCC0N: return "DCC0N";
  case ControlRow::DCC1: return "DCC1";
  case ControlRow::DCC1N: return "DCC1N";
  case ControlRow::DCC0N_T0: return "DCC0N_T0";
  case ControlRow::DCC1N_T1: return "DCC1N_T1";
  case ControlRow::T2_T3: return "T2_T3";
  case ControlRow::T0_T3: return "T0_T3";
  case ControlRow::T0_T1_T2: return "T0_T1_T2";
  case ControlRow::T0_T1_T3: return "T0_T1_T3";
  case ControlRow::DCC0_T1_T3: return "DCC0_T1_T3";
  case ControlRow::DCC1_T0_T2: return "DCC1_T0_T2";
  case ControlRow::C_0: return "C_0";
  case ControlRow::C_1: return "C_1";
  case ControlRow::NUM_CONTROL_ROWS: break;
  }
  llvm_unreachable("unhandled control row");
}

llvm::StringRef mlir::bits::getRowOpName(RowOpKind kind) {
  switch (kind) {
  case RowOpKind::And: return "row_and";
  case RowOpKind::Or: return "row_or";
  case RowOpKind::Not: return "row_not";
  case RowOpKind::Maj3: return "maj3";
  case RowOpKind::Xor: return "row_xor";
  case RowOpKind::Xnor: return "row_xnor";
  case RowOpKind::FullAdd: return "row_add";
  case RowOpKind::FullAddLoaded: return "row_add_loaded";
  case RowOpKind::FullSub: return "row_sub";
  case RowOpKind::Relu: return "row_relu";
  case RowOpKind::MinMax: return "row_minmax";
  case RowOpKind::RangeScan00: return "row_scan_00";
  case RowOpKind::RangeScan01: return "row_scan_01";
  case RowOpKind::RangeScan10: return "row_scan_10";
  case RowOpKind::RangeScan11: return "row_scan_11";
  case RowOpKind::NUM_KINDS: break;
  }
  llvm_unreachable("unhandled row-op kind");
}

RowOpKind mlir::bits::getRangeScanKind(bool lowerBit, bool upperBit) {
  if (!lowerBit)
    return upperBit ? RowOpKind::RangeScan01 : RowOpKind::RangeScan00;
  return upperBit ? RowOpKind::RangeScan11 : RowOpKind::RangeScan10;
}

std::string RowOperand::str() const {
  switch (kind) {
  case Kind::Control:
    return getControlRowName(getControlRow()).str();
  case Kind::Input:
    return "in" + std::to_string(index);
  case Kind::Output:
    return "out" + std::to_string(index);
  case Kind::Temp:
    return "tmp" + std::to_string(index);
  }
  llvm_unreachable("unhandled operand kind");
}

std::string RowOpInstr::str() const {
  if (isAp)
    return "AP (" + dst.str() + ")";
  return "AAP(" + dst.str() + " <- " + src.str() + ")";
}

//===----------------------------------------------------------------------===//
// Reference counts
//===----------------------------------------------------------------------===//

unsigned mlir::bits::getReferenceCommands(PudBackend backend, RowOpKind kind) {
  // Command counts of the hand-written expansions in rowop_trace_player.cc.
  //
  // These are the numbers the legacy cost table was derived from. Two carry
  // caveats worth knowing before treating a difference as a regression:
  //
  //  * row_xor's 7 is inferred from the XNOR sequence by symmetry -- the
  //    simulator has no expansion for XOR.
  //  * the range-scan step's 4 is a timing skeleton, not a working sequence:
  //    expandRangeScan's second per-bit command writes C_0 into the (!DCC1, T1)
  //    group, which drives DCC1 -- the ">= lower" accumulator -- to 1 on every
  //    iteration, so that chain never advances and the emitted mask depends on
  //    the "<= upper" chain alone. The simulator has no functional semantics for
  //    row ops, so nothing downstream could have caught it.
  if (backend != PudBackend::Simdram)
    return 0;
  switch (kind) {
  case RowOpKind::And: return 4;          // emitRowAnd
  case RowOpKind::Or: return 4;
  case RowOpKind::Xnor: return 7;         // expandXnor
  case RowOpKind::Xor: return 7;          // inferred, see above
  case RowOpKind::FullAdd: return 8;      // expandAddSimdram, per bit
  case RowOpKind::FullAddLoaded: return 9;// emitRowAdd
  case RowOpKind::FullSub: return 9;      // expandSubSimdram, per bit
  case RowOpKind::Relu: return 4;         // expandRelu reuses emitRowAnd
  case RowOpKind::MinMax: return 13;      // expandMin / expandMax, per bit
  case RowOpKind::RangeScan00:
  case RowOpKind::RangeScan01:
  case RowOpKind::RangeScan10:
  case RowOpKind::RangeScan11:
    return 4;                             // timing skeleton, see above
  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// Library
//===----------------------------------------------------------------------===//

RowOpLibrary &RowOpLibrary::get() {
  static RowOpLibrary library;
  return library;
}

RowOpLibrary::Entry &RowOpLibrary::entryFor(PudBackend backend,
                                            RowOpKind kind) {
  return cache[static_cast<unsigned>(backend)][static_cast<unsigned>(kind)];
}

const RowOpTemplate *RowOpLibrary::getTemplate(PudBackend backend,
                                               RowOpKind kind) {
  // lime only has an architecture definition for SIMDRAM; PRADA and FCDRAM stay
  // on the legacy cost table until one is added.
  if (backend != PudBackend::Simdram)
    return nullptr;

  Entry &entry = entryFor(backend, kind);
  if (entry.failed)
    return nullptr;
  if (!entry.compiled) {
    entry.compiled = true;
    if (!compileTemplate(kind, entry.tmpl)) {
      entry.failed = true;
      return nullptr;
    }
  }
  return &entry.tmpl;
}

std::optional<RowOpCost> RowOpLibrary::getCost(PudBackend backend,
                                               RowOpKind kind) {
  const RowOpTemplate *tmpl = getTemplate(backend, kind);
  if (!tmpl)
    return std::nullopt;
  return RowOpCost{tmpl->numAAP, tmpl->numAP};
}

namespace {
/// Pads to a fixed column width. Written by hand rather than with formatv
/// because this file is compiled with RTTI on (mockturtle needs it) while LLVM
/// is not, and formatv's adapters would emit type information that cannot be
/// linked against.
void pad(llvm::raw_ostream &os, llvm::StringRef text, unsigned width,
         bool left = true) {
  unsigned fill = text.size() < width ? width - text.size() : 0;
  if (!left)
    os.indent(fill);
  os << text;
  if (left)
    os.indent(fill);
}
void padNum(llvm::raw_ostream &os, int64_t value, unsigned width) {
  pad(os, std::to_string(value), width, /*left=*/false);
}
} // namespace

std::string RowOpLibrary::renderSanityTable(PudBackend backend) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "row-op templates for backend '" << getBackendName(backend) << "'\n";
  pad(os, "primitive", 17);
  pad(os, "lime", 7, false);
  pad(os, "AAP", 7, false);
  pad(os, "AP", 7, false);
  pad(os, "ref", 7, false);
  pad(os, "delta", 10, false);
  os << "  validated\n";
  for (unsigned i = 0; i < static_cast<unsigned>(RowOpKind::NUM_KINDS); i++) {
    auto kind = static_cast<RowOpKind>(i);
    const RowOpTemplate *tmpl = getTemplate(backend, kind);
    pad(os, getRowOpName(kind), 17);
    if (!tmpl) {
      pad(os, "n/a", 7, false);
      os << "\n";
      continue;
    }
    unsigned ref = getReferenceCommands(backend, kind);
    padNum(os, tmpl->totalCommands(), 7);
    padNum(os, tmpl->numAAP, 7);
    padNum(os, tmpl->numAP, 7);
    pad(os, ref ? std::to_string(ref) : "-", 7, false);
    std::string delta;
    if (!ref)
      delta = "-";
    else if (tmpl->totalCommands() == ref)
      delta = "match";
    else {
      int diff = static_cast<int>(tmpl->totalCommands()) - static_cast<int>(ref);
      delta = (diff > 0 ? "+" : "") + std::to_string(diff);
    }
    pad(os, delta, 10, false);
    os << "  " << (tmpl->validated ? "yes" : "NO") << "\n";
  }
  // the instruction streams themselves, so a reader can check a template
  // against the hand-written expansion command by command
  for (unsigned i = 0; i < static_cast<unsigned>(RowOpKind::NUM_KINDS); i++) {
    auto kind = static_cast<RowOpKind>(i);
    const RowOpTemplate *tmpl = getTemplate(backend, kind);
    if (!tmpl)
      continue;
    os << "\n" << getRowOpName(kind) << ":\n";
    for (const RowOpInstr &instr : tmpl->instrs)
      os << "  " << instr.str() << "\n";
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Recipes
//===----------------------------------------------------------------------===//

std::optional<int64_t>
RowOpRecipe::totalCommands(PudBackend backend) const {
  int64_t total = fixedAAP + fixedAP;
  for (const Step &step : steps) {
    auto cost = RowOpLibrary::get().getCost(backend, step.kind);
    if (!cost)
      return std::nullopt;
    total += static_cast<int64_t>(cost->total()) * step.count;
  }
  return total;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForAdd(int64_t width,
                                                       bool widening) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::FullAdd, width});
  // one command seeds the carry to zero; a widening add stores the carry-out as
  // the extra result bit rather than dropping it
  recipe.fixedAAP = widening ? 2 : 1;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForSub(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::FullSub, width});
  // the carry seeds to one: the +1 of ~rhs + 1
  recipe.fixedAAP = 1;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForMul(int64_t lhsWidth,
                                                       int64_t rhsWidth) {
  if (lhsWidth <= 0 || rhsWidth <= 0)
    return std::nullopt;
  // Shift-add over the bit-sliced operands, following expandMulSimdram: each
  // partial product is one row AND, and each accumulation one full adder whose
  // carry comes from a partial-product row rather than a resident cell.
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::And, lhsWidth * (rhsWidth + 1)});
  recipe.steps.push_back(
      {RowOpKind::FullAddLoaded, lhsWidth * rhsWidth - 1});
  recipe.fixedAAP = 1;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForAnd(int64_t width,
                                                       bool maskedZero) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::And, width});
  // a ReLU-style masked AND builds ~sign once, through the dual-contact cell,
  // and parks it in a row the per-bit AND does not touch
  recipe.fixedAAP = maskedZero ? 2 : 0;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForOr(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::Or, width});
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForXnor(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::Xnor, width});
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForRangeScan(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  // Priced off the scanned column's width: the result is a single mask row and
  // says nothing about the work. Every branch of the per-bit case analysis
  // compiles to the same command count, so the scanned constants do not enter
  // the cost -- only the choice of which of the four templates runs.
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::RangeScan11, width});
  // three commands seed both chains, two fold them into the output mask
  recipe.fixedAAP = 5;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForMinMax(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::MinMax, width});
  // seeding the verdict row
  recipe.fixedAAP = 2;
  return recipe;
}
