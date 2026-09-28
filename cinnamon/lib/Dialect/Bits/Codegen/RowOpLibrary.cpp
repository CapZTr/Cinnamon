//===- RowOpLibrary.cpp - row-op primitives compiled by lime --------------===//

#include "cinm-mlir/Dialect/Bits/Codegen/RowOpLibrary.h"
#include "PrimitiveFile.h"

#include "eggmock.hpp"
#include "gp.h"
#include "mockturtle/algorithms/cleanup.hpp"
#include "mockturtle/networks/mig.hpp"
#include "utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <cstring>
#include <map>
#include <string>
#include <utility>

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

/// One bit of the comparison `a > b`, least significant bit upwards:
///
///     g' = maj(a, !b, g)
///
///   a and b equal  -> maj(x, !x, g) = g, the lower bits' verdict carries
///   a=1, b=0       -> maj(1, 1, g)  = 1
///   a=0, b=1       -> maj(0, 0, g)  = 0
///
/// The whole comparator is that one majority gate per bit.
MIG buildGreaterStep() {
  MIG n;
  auto a = n.create_pi(), b = n.create_pi(), g = n.create_pi();
  n.create_po(n.create_maj(a, !b, g));
  return n;
}

/// One bit of `g ? x : y`, the select a min or a max ends with.
MIG buildSelect() {
  MIG n;
  auto x = n.create_pi(), y = n.create_pi(), g = n.create_pi();
  auto f = n.get_constant(false), t = n.get_constant(true);
  n.create_po(n.create_maj(n.create_maj(g, x, f), n.create_maj(!g, y, f), t));
  return n;
}

/// One BitWeaving range-scan bit step:
///   R' = maj(R,  v, !lower[j])    ">= lower" chain
///   S' = maj(S, !v,  upper[j])    "<= upper" chain
///
/// `maj(acc, x, k)` is OR when k is one and AND when k is zero, so k is the
/// selector, and the two chains do not read it the same way round. Going from
/// the least significant bit up:
///
///   lower[j] == 0: a one in the value settles ">= lower" whatever lies below,
///                  a zero leaves the verdict to the lower bits -- so OR,
///                  selector one, the bound's bit INVERTED.
///   lower[j] == 1: a zero in the value settles it the other way -- AND.
///   upper[j] == 1: OR, and the chain reads !v, which puts its selector the
///                  same way round as the bound's bit.
///
/// The lower chain's selector was the bound's bit uninverted here until
/// 2026-08-17, which made the ">= lower" half of every scan compute the
/// complement of the intended comparison. Nothing had caught it because nothing
/// consumed these templates yet -- `getRangeScanKind` had no callers -- and it
/// costs no commands either way, both selectors being constant rows.
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
  n.create_po(n.create_maj(r, v, n.get_constant(!lowerBit)));
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
  case RowOpKind::GreaterStep:
    return buildGreaterStep();
  case RowOpKind::Select:
    return buildSelect();
  case RowOpKind::RangeScan00:
    return buildRangeScan(false, false);
  case RowOpKind::RangeScan01:
    return buildRangeScan(false, true);
  case RowOpKind::RangeScan10:
    return buildRangeScan(true, false);
  case RowOpKind::RangeScan11:
    return buildRangeScan(true, true);
  case RowOpKind::ScanFold:
    return buildAnd();
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
// A model of the substrate
//===----------------------------------------------------------------------===//

/// One physical row an activation drives.
struct RowMember {
  enum class Kind { Compute, Cell, Constant } kind;
  /// Compute row number, dual-contact cell number, or constant value.
  int index;
  /// Set for a cell's complementing wordline, which reads and writes inverted.
  bool inverted;
};

/// The rows a control-row name activates. The compound names come from the one
/// table the lime translation also reads, so the two cannot disagree about what
/// a group is.
llvm::SmallVector<RowMember, 3> membersOf(ControlRow row) {
  switch (row) {
  case ControlRow::T0:
  case ControlRow::T1:
  case ControlRow::T2:
  case ControlRow::T3:
    return {{RowMember::Kind::Compute,
             static_cast<int>(row) - static_cast<int>(ControlRow::T0), false}};
  case ControlRow::DCC0:
    return {{RowMember::Kind::Cell, 0, false}};
  case ControlRow::DCC0N:
    return {{RowMember::Kind::Cell, 0, true}};
  case ControlRow::DCC1:
    return {{RowMember::Kind::Cell, 1, false}};
  case ControlRow::DCC1N:
    return {{RowMember::Kind::Cell, 1, true}};
  case ControlRow::C_0:
    return {{RowMember::Kind::Constant, 0, false}};
  case ControlRow::C_1:
    return {{RowMember::Kind::Constant, 1, false}};
  default:
    break;
  }
  for (const GroupPattern &group : getSimdramGroups()) {
    if (group.row != row)
      continue;
    llvm::SmallVector<RowMember, 3> members;
    for (auto [i, cell] : llvm::enumerate(group.cells))
      members.push_back({llvm::StringRef(cell.first) == "DCC"
                             ? RowMember::Kind::Cell
                             : RowMember::Kind::Compute,
                         static_cast<int>(cell.second), group.inverted[i]});
    return members;
  }
  llvm::report_fatal_error("control row is not in the substrate's group table");
}

/// The state a row-op sequence runs against: the four compute rows, the two
/// dual-contact cells, and the data rows a template's operands name.
///
/// A dual-contact cell holds one value and presents it on two wordlines, so
/// reading or writing through the complementing one inverts. That is the whole
/// of why an inversion costs no command on this substrate, and getting it right
/// is most of what this model is for.
class Substrate {
public:
  Substrate(bool computeInit[4], bool cellInit[2], bool dataInit) {
    for (int i = 0; i < 4; i++)
      compute[i] = computeInit[i];
    for (int i = 0; i < 2; i++)
      cell[i] = cellInit[i];
    dataDefault = dataInit;
  }

  /// Runs one instruction. Returns failure if it names something this substrate
  /// cannot do, which is a bug in the template rather than a wrong answer.
  bool run(const RowOpInstr &instr) {
    if (instr.isAp)
      return activate(instr.dst).has_value();
    std::optional<bool> value = read(instr.src);
    if (!value)
      return false;
    return write(instr.dst, *value);
  }

  std::optional<bool> read(const RowOperand &operand) {
    if (!operand.isControl())
      return dataRow(operand);
    auto members = membersOf(operand.getControlRow());
    if (members.size() == 3)
      return activate(operand); // a triple-row activation, read as its majority
    if (members.size() != 1)
      return std::nullopt;      // a row pair has no value of its own
    return readMember(members[0]);
  }

private:
  /// Activates a row group: the three rows drive the bitline together, the
  /// majority wins, and every one of them is left holding it.
  std::optional<bool> activate(const RowOperand &operand) {
    if (!operand.isControl())
      return std::nullopt;
    auto members = membersOf(operand.getControlRow());
    if (members.size() != 3)
      return std::nullopt;
    int ones = 0;
    for (const RowMember &member : members) {
      std::optional<bool> bit = readMember(member);
      if (!bit)
        return std::nullopt;
      ones += *bit;
    }
    bool majority = ones >= 2;
    for (const RowMember &member : members)
      if (!writeMember(member, majority))
        return std::nullopt;
    return majority;
  }

  bool write(const RowOperand &operand, bool value) {
    if (!operand.isControl()) {
      dataRow(operand) = value;
      return true;
    }
    for (const RowMember &member : membersOf(operand.getControlRow()))
      if (!writeMember(member, value))
        return false;
    return true;
  }

  std::optional<bool> readMember(const RowMember &member) {
    switch (member.kind) {
    case RowMember::Kind::Compute:
      return compute[member.index];
    case RowMember::Kind::Cell:
      return member.inverted ? !cell[member.index] : cell[member.index];
    case RowMember::Kind::Constant:
      return member.index != 0;
    }
    return std::nullopt;
  }

  bool writeMember(const RowMember &member, bool value) {
    switch (member.kind) {
    case RowMember::Kind::Compute:
      compute[member.index] = value;
      return true;
    case RowMember::Kind::Cell:
      cell[member.index] = member.inverted ? !value : value;
      return true;
    case RowMember::Kind::Constant:
      return false; // a row held at a constant is not a destination
    }
    return false;
  }

  bool &dataRow(const RowOperand &operand) {
    auto key = std::make_pair(static_cast<int>(operand.kind), operand.index);
    auto [it, inserted] = data.try_emplace(key, dataDefault);
    return it->second;
  }

  bool compute[4];
  bool cell[2];
  bool dataDefault;
  std::map<std::pair<int, int>, bool> data;
};

/// What each primitive computes, spelled the same way its MIG builder above
/// spells it. This is the reference the model checks a sequence against, so it
/// deliberately restates the majority expressions rather than the arithmetic
/// they add up to.
llvm::SmallVector<bool, 2> expectedOutputs(RowOpKind kind,
                                           llvm::ArrayRef<bool> in) {
  auto maj = [](bool a, bool b, bool c) { return (a && b) || (a && c) || (b && c); };
  switch (kind) {
  case RowOpKind::And:
    return {maj(in[0], in[1], false)};
  case RowOpKind::Or:
    return {maj(in[0], in[1], true)};
  case RowOpKind::Not:
    return {!in[0]};
  case RowOpKind::Maj3:
    return {maj(in[0], in[1], in[2])};
  case RowOpKind::Xor:
    return {maj(maj(in[0], in[1], true), !maj(in[0], in[1], false), false)};
  case RowOpKind::Xnor:
    return {maj(maj(in[0], !in[1], true), maj(!in[0], in[1], true), false)};
  case RowOpKind::FullAdd:
  case RowOpKind::FullAddLoaded: {
    bool cout = maj(in[0], in[1], in[2]);
    return {maj(in[1], !cout, maj(in[0], !in[1], in[2])), cout};
  }
  case RowOpKind::FullSub: {
    bool cout = maj(in[0], !in[1], in[2]);
    return {maj(!in[1], !cout, maj(in[0], in[1], in[2])), cout};
  }
  case RowOpKind::Relu:
    return {maj(in[0], !in[1], false)};
  case RowOpKind::MinMax: {
    auto ite = [&](bool s, bool x, bool y) {
      return maj(maj(s, x, false), maj(!s, y, false), true);
    };
    return {ite(in[2], in[1], in[0]), ite(in[2], in[0], in[1])};
  }
  case RowOpKind::RangeScan00:
  case RowOpKind::RangeScan01:
  case RowOpKind::RangeScan10:
  case RowOpKind::RangeScan11: {
    bool lower = kind == RowOpKind::RangeScan10 || kind == RowOpKind::RangeScan11;
    bool upper = kind == RowOpKind::RangeScan01 || kind == RowOpKind::RangeScan11;
    return {maj(in[1], in[0], !lower), maj(in[2], !in[0], upper)};
  }
  case RowOpKind::ScanFold:
    return {maj(in[0], in[1], false)};
  case RowOpKind::GreaterStep:
    return {maj(in[0], !in[1], in[2])};
  case RowOpKind::Select:
    return {maj(maj(in[2], in[0], false), maj(!in[2], in[1], false), true)};
  case RowOpKind::NUM_KINDS:
    break;
  }
  llvm_unreachable("unhandled row-op kind");
}

/// Runs `tmpl` over every input assignment and every initial state of the
/// compute rows and cells, checking it reproduces the primitive.
///
/// Sweeping the initial state is the part that matters: a sequence that happens
/// to work because a compute row was left at zero would pass a single-state
/// check and then fail on the second bit of the first operand.
bool simulate(RowOpKind kind, const RowOpTemplate &tmpl) {
  if (tmpl.inputBindings.size() != tmpl.numInputs ||
      tmpl.outputBindings.size() != tmpl.numOutputs)
    return false;

  unsigned numInputs = tmpl.numInputs;
  for (unsigned assignment = 0; assignment < (1u << numInputs); assignment++) {
    llvm::SmallVector<bool, 3> inputs;
    for (unsigned i = 0; i < numInputs; i++)
      inputs.push_back((assignment >> i) & 1);
    auto expected = expectedOutputs(kind, inputs);
    if (expected.size() != tmpl.numOutputs)
      return false;

    for (unsigned state = 0; state < 64; state++) {
      for (bool dataInit : {false, true}) {
        bool compute[4], cell[2];
        for (int i = 0; i < 4; i++)
          compute[i] = (state >> i) & 1;
        for (int i = 0; i < 2; i++)
          cell[i] = (state >> (4 + i)) & 1;

        Substrate substrate(compute, cell, dataInit);
        // The caller's job before the sequence runs: every input is in the row
        // the template says it reads it from.
        for (unsigned i = 0; i < numInputs; i++) {
          RowOpInstr load;
          load.src = RowOperand::control(inputs[i] ? ControlRow::C_1
                                                  : ControlRow::C_0);
          load.dst = tmpl.inputBindings[i];
          if (!substrate.run(load))
            return false;
        }
        for (const RowOpInstr &instr : tmpl.instrs)
          if (!substrate.run(instr))
            return false;
        for (unsigned i = 0; i < tmpl.numOutputs; i++) {
          std::optional<bool> got = substrate.read(tmpl.outputBindings[i]);
          if (!got || *got != expected[i])
            return false;
        }
      }
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Hand-written templates
//===----------------------------------------------------------------------===//

/// Shorthands for spelling a sequence out.
RowOpInstr aap(RowOperand dst, RowOperand src) { return {false, src, dst}; }
RowOpInstr ap(ControlRow group) {
  return {true, RowOperand(), RowOperand::control(group)};
}
RowOperand ctrl(ControlRow row) { return RowOperand::control(row); }

/// XOR and XNOR share one seven-command shape, differing only in which constant
/// seeds the pair of majorities and which one closes them:
///
///   XNOR = (a | !b) & (!a | b)      XOR = (a & !b) | (!a & b)
///
/// Both complements come out of the dual-contact cells for free, and the two
/// inner majorities read disjoint rows, so they run back to back without either
/// clobbering what the other still needs:
///
///   1. AAP(DCC0N_T0 <- a)   cell 0 holds !a, so row DCC0 reads !a;  T0 = a
///   2. AAP(DCC1N_T1 <- b)   cell 1 holds !b, so row DCC1 reads !b;  T1 = b
///   3. AAP(T2_T3    <- seed)
///   4. AP (DCC0_T1_T3)      maj(!a, b, seed)   -> DCC0, T1, T3
///   5. AP (DCC1_T0_T2)      maj(!b, a, seed)   -> DCC1, T0, T2
///   6. AAP(T2 <- close)     T0 still holds step 5's result
///   7. AAP(out <- T0_T1_T2) maj(step 5, step 4, close)
///
/// lime schedules XNOR instead as the complement of an XOR, which costs two more
/// commands, and XOR one more; on a BNN layer that is 128 commands per tile per
/// bank of pure inversion. Hence writing them out.
RowOpTemplate buildEquality(bool negated) {
  ControlRow seed = negated ? ControlRow::C_1 : ControlRow::C_0;
  ControlRow close = negated ? ControlRow::C_0 : ControlRow::C_1;
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::DCC0N_T0), RowOperand::input(0)),
      aap(ctrl(ControlRow::DCC1N_T1), RowOperand::input(1)),
      aap(ctrl(ControlRow::T2_T3), ctrl(seed)),
      ap(ControlRow::DCC0_T1_T3),
      ap(ControlRow::DCC1_T0_T2),
      aap(ctrl(ControlRow::T2), ctrl(close)),
      aap(RowOperand::output(0), ctrl(ControlRow::T0_T1_T2)),
  };
  tmpl.numInputs = 2;
  tmpl.numOutputs = 1;
  tmpl.inputBindings = {RowOperand::input(0), RowOperand::input(1)};
  tmpl.outputBindings = {RowOperand::output(0)};
  return tmpl;
}

/// Full subtractor, nine commands, carry-in and carry-out resident in cell 1.
///
/// The subtractor is the adder with the subtrahend complemented throughout:
///
///   adder       cout = maj(a,  b, c)   m = maj(a, !b, c)   sum  = maj( b, !cout, m)
///   subtractor  cout = maj(a, !b, c)   m = maj(a,  b, c)   diff = maj(!b, !cout, m)
///
/// so it is lime's eight-command adder with b and !b exchanged at every read. The
/// exchange is what costs the ninth command: the adder gets b for free from the
/// row it loads it into and !b from the cell it parks it in, and needs each once;
/// the subtractor needs !b before the cell has been written and b after, so the
/// cell has to be turned over (step 6) instead of loaded once.
///
///   1. AAP(DCC0N <- b)      cell 0 holds !b, so row DCC0 reads !b
///   2. AAP(T0_T3 <- a)
///   3. AAP(T2 <- DCC0)      T2 = !b
///   4. AAP(T1 <- DCC1)      T1 = the carry-in
///   5. AP (DCC1_T0_T2)      maj(c, a, !b) = carry-out -> DCC1, T0, T2
///   6. AAP(DCC0N_T0 <- DCC0)  turn the cell over: DCC0 now reads b, T0 = !b
///   7. AP (DCC0_T1_T3)      maj(b, c, a) = m -> DCC0, T1, T3
///   8. AAP(T2 <- DCC1N)     T2 = !carry-out
///   9. AAP(out0 <- T0_T1_T2)  maj(!b, m, !carry-out) = the difference
///
/// lime schedules ten. Nine is also what the simulator's expandSubSimdram costs,
/// down to the same seven-AAP two-AP split, though it reaches it over Ambit's row
/// groups rather than these.
RowOpTemplate buildSubtractor() {
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::DCC0N), RowOperand::input(1)),
      aap(ctrl(ControlRow::T0_T3), RowOperand::input(0)),
      aap(ctrl(ControlRow::T2), ctrl(ControlRow::DCC0)),
      aap(ctrl(ControlRow::T1), ctrl(ControlRow::DCC1)),
      ap(ControlRow::DCC1_T0_T2),
      aap(ctrl(ControlRow::DCC0N_T0), ctrl(ControlRow::DCC0)),
      ap(ControlRow::DCC0_T1_T3),
      aap(ctrl(ControlRow::T2), ctrl(ControlRow::DCC1N)),
      aap(RowOperand::output(0), ctrl(ControlRow::T0_T1_T2)),
  };
  tmpl.numInputs = 3;
  tmpl.numOutputs = 2;
  // The carry-in is read from cell 1 and the carry-out is left there, which is
  // what makes the borrow chain free to keep across bit positions.
  tmpl.inputBindings = {RowOperand::input(0), RowOperand::input(1),
                        ctrl(ControlRow::DCC1)};
  tmpl.outputBindings = {RowOperand::output(0), ctrl(ControlRow::DCC1)};
  return tmpl;
}

/// One BitWeaving range-scan bit step, five commands.
///
///   R' = maj(R,  v, !lower[j])     ">= lower"
///   S' = maj(S, !v,  upper[j])     "<= upper"
///
/// The two chains want opposite senses of the value, and there are only two
/// dual-contact cells -- both already holding a chain -- so there is nowhere to
/// put `v` to get `!v` for free. Carry the second chain COMPLEMENTED instead:
/// complementing a majority complements all three of its inputs, so
///
///   !S' = maj(!S, v, !upper[j])
///
/// and now both chains read the same `v`, which one command puts in both rows
/// they read it from. The cell holds !S and row DCC1N reads S, which is what the
/// bindings say and what the fold reads; nothing downstream sees the difference.
///
///   1. AAP(T2_T3 <- v)        one command, both the rows the chains read
///   2. AAP(T1 <- !lower[j])
///   3. AP (DCC0_T1_T3)        maj(R, !lower, v) -> DCC0, T1, T3
///   4. AAP(T0 <- !upper[j])
///   5. AP (DCC1_T0_T2)        maj(!S, !upper, v) -> DCC1, T0, T2
///
/// Step 3 destroys T3, which is why `v` is in T2 as well; nothing else in the
/// sequence writes a row a later step reads.
///
/// lime schedules seven, spending two of them moving R out of cell 0 to make
/// room for `v` and then putting it back. Worth writing out for more than the
/// two commands: lime's four variants do not share a command mix -- two are five
/// AAPs and two APs, two are six and one -- and the trace player cannot know a
/// bound's bits, so there is no sequence it could replay that would agree with
/// all four. These five are the same shape whatever the bounds are.
RowOpTemplate buildRangeScanStep(bool lowerBit, bool upperBit) {
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::T2_T3), RowOperand::input(0)),
      aap(ctrl(ControlRow::T1),
          ctrl(lowerBit ? ControlRow::C_0 : ControlRow::C_1)),
      ap(ControlRow::DCC0_T1_T3),
      aap(ctrl(ControlRow::T0),
          ctrl(upperBit ? ControlRow::C_0 : ControlRow::C_1)),
      ap(ControlRow::DCC1_T0_T2),
  };
  tmpl.numInputs = 3;
  tmpl.numOutputs = 2;
  tmpl.inputBindings = {RowOperand::input(0), ctrl(ControlRow::DCC0),
                        ctrl(ControlRow::DCC1N)};
  tmpl.outputBindings = {ctrl(ControlRow::DCC0), ctrl(ControlRow::DCC1N)};
  return tmpl;
}

/// Folds the two range-scan chains into the match mask: `mask = R AND S`.
///
///   1. AAP(T3 <- DCC1N)            T3 = the "<= upper" chain
///   2. AAP(T1 <- C_0)
///   3. AAP(out0 <- DCC0_T1_T3)     maj(R, 0, S) = R AND S
///
/// Three commands, and three is the floor: the group this reads the "<= lower"
/// chain out of needs T1 and T3 set, and no row pair the substrate wires covers
/// both. Written out because lime would be given a three-input majority with no
/// way to know two of its operands are already sitting in cells, and would pay a
/// command restating one of them.
///
/// It reads the second chain from the cell's complementing contact because that
/// is the side the chain is kept on -- see buildRangeScanStep.
RowOpTemplate buildScanFold() {
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::T3), ctrl(ControlRow::DCC1N)),
      aap(ctrl(ControlRow::T1), ctrl(ControlRow::C_0)),
      aap(RowOperand::output(0), ctrl(ControlRow::DCC0_T1_T3)),
  };
  tmpl.numInputs = 2;
  tmpl.numOutputs = 1;
  tmpl.inputBindings = {ctrl(ControlRow::DCC0), ctrl(ControlRow::DCC1N)};
  tmpl.outputBindings = {RowOperand::output(0)};
  return tmpl;
}

/// One bit of the comparison a min or a max starts with, three commands.
///
///   1. AAP(T0 <- a)
///   2. AAP(DCC1N_T1 <- b)   cell 1 holds !b, so row DCC1 reads !b
///   3. AP (DCC1_T0_T2)      maj(!b, a, g) -> DCC1, T0, T2
///
/// The verdict lives in T2 and is never written by steps 1 and 2, so the chain
/// costs nothing to keep across bit positions -- a compute row does here what a
/// dual-contact cell does for a carry. It is in a compute row rather than a cell
/// because it is read three times per bit and complemented none, which is the
/// opposite of what a cell is for; the complement is wanted once, later, and the
/// select pays a single command to move it into a cell for that.
RowOpTemplate buildGreater() {
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::T0), RowOperand::input(0)),
      aap(ctrl(ControlRow::DCC1N_T1), RowOperand::input(1)),
      ap(ControlRow::DCC1_T0_T2),
  };
  tmpl.numInputs = 3;
  tmpl.numOutputs = 1;
  tmpl.inputBindings = {RowOperand::input(0), RowOperand::input(1),
                        ctrl(ControlRow::T2)};
  tmpl.outputBindings = {ctrl(ControlRow::T2)};
  return tmpl;
}

/// One bit of `g ? x : y`, ten commands, with g resident in cell 0.
///
///   1. AAP(T1 <- DCC0)          T1 = g
///   2. AAP(T0 <- x)
///   3. AAP(T2 <- C_0)
///   4. AAP(T3 <- T0_T1_T2)      T3 = maj(x, g, 0) = g AND x
///   5. AAP(T1 <- DCC0N)         T1 = !g, free from the same cell
///   6. AAP(T0 <- y)
///   7. AAP(T2 <- C_0)
///   8. AP (T0_T1_T2)            maj(y, !g, 0) = !g AND y
///   9. AAP(T1 <- C_1)
///  10. AAP(out0 <- T0_T1_T3)    maj(!g AND y, 1, g AND x) = the selection
///
/// Nothing in the sequence writes cell 0, so one select follows another over a
/// whole operand on a single copy of g. That is what makes the pair of chains
/// cost 13n + 2 rather than 13n + 2 per bit.
RowOpTemplate buildSelectProgram() {
  RowOpTemplate tmpl;
  tmpl.instrs = {
      aap(ctrl(ControlRow::T1), ctrl(ControlRow::DCC0)),
      aap(ctrl(ControlRow::T0), RowOperand::input(0)),
      aap(ctrl(ControlRow::T2), ctrl(ControlRow::C_0)),
      aap(ctrl(ControlRow::T3), ctrl(ControlRow::T0_T1_T2)),
      aap(ctrl(ControlRow::T1), ctrl(ControlRow::DCC0N)),
      aap(ctrl(ControlRow::T0), RowOperand::input(1)),
      aap(ctrl(ControlRow::T2), ctrl(ControlRow::C_0)),
      ap(ControlRow::T0_T1_T2),
      aap(ctrl(ControlRow::T1), ctrl(ControlRow::C_1)),
      aap(RowOperand::output(0), ctrl(ControlRow::T0_T1_T3)),
  };
  tmpl.numInputs = 3;
  tmpl.numOutputs = 1;
  tmpl.inputBindings = {RowOperand::input(0), RowOperand::input(1),
                        ctrl(ControlRow::DCC0)};
  tmpl.outputBindings = {RowOperand::output(0)};
  return tmpl;
}

/// The primitives lime's scheduler does not compile to the shortest known
/// program. Everything absent from here is lime's.
std::optional<RowOpTemplate> getHandWrittenTemplate(PudBackend backend,
                                                    RowOpKind kind) {
  if (backend != PudBackend::Simdram)
    return std::nullopt;
  switch (kind) {
  case RowOpKind::Xor:
    return buildEquality(/*negated=*/false);
  case RowOpKind::Xnor:
    return buildEquality(/*negated=*/true);
  case RowOpKind::FullSub:
    return buildSubtractor();
  case RowOpKind::RangeScan00:
    return buildRangeScanStep(/*lowerBit=*/false, /*upperBit=*/false);
  case RowOpKind::RangeScan01:
    return buildRangeScanStep(/*lowerBit=*/false, /*upperBit=*/true);
  case RowOpKind::RangeScan10:
    return buildRangeScanStep(/*lowerBit=*/true, /*upperBit=*/false);
  case RowOpKind::RangeScan11:
    return buildRangeScanStep(/*lowerBit=*/true, /*upperBit=*/true);
  case RowOpKind::ScanFold:
    return buildScanFold();
  case RowOpKind::GreaterStep:
    return buildGreater();
  case RowOpKind::Select:
    return buildSelectProgram();
  default:
    return std::nullopt;
  }
}

//===----------------------------------------------------------------------===//
// Compilation
//===----------------------------------------------------------------------===//

bool compileTemplate(RowOpKind kind, RowOpTemplate &out) {
  // A definition from --pud-primitives replaces the built-in network and its
  // hints. The copy matters: a network shares its storage with the one it was
  // copied from, and preoptimize rewrites in place.
  const PrimitiveDefinition *def = lookupFilePrimitive(kind);
  MIG mig = def ? mockturtle::cleanup_dangling(def->mig) : buildMig(kind);
  if (def) {
    // The primitive's meaning is fixed -- it is what the substrate check
    // compares against -- so the file's inputs and outputs have to line up.
    MIG builtin = buildMig(kind);
    if (mig.num_pis() != builtin.num_pis() || mig.num_pos() != builtin.num_pos())
      llvm::report_fatal_error(
          llvm::Twine("--pud-primitives=") + primitiveFilePath() + ": @" +
          getRowOpName(kind) + " takes " + llvm::Twine(mig.num_pis()) +
          " inputs and gives " + llvm::Twine(mig.num_pos()) +
          " outputs; the primitive has " + llvm::Twine(builtin.num_pis()) +
          " and " + llvm::Twine(builtin.num_pos()));
  }
  if (!def || def->preoptimize)
    preoptimize(mig);
  out.fromFile = def != nullptr;

  LoopCarry lc;
  if (def) {
    for (const PrimitiveHint &h : def->inputs)
      lc.inputs.push_back({h.index, h.cellType, h.cellIndex});
    for (const PrimitiveHint &h : def->outputs)
      lc.outputs.push_back({h.index, h.cellType, h.cellIndex});
  } else {
    lc = getLoopCarry(kind);
  }
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

  if (def) {
    // The file's own count is the requirement; the built-in one is only
    // reported, since a different network is exactly what a file is for.
    auto builtin = getRequiredCommands(kind);
    if (def->commands && out.totalCommands() != *def->commands)
      llvm::report_fatal_error(llvm::Twine("row-op template '") +
                               getRowOpName(kind) + "' from " +
                               primitiveFilePath() + " compiled to " +
                               llvm::Twine(out.totalCommands()) +
                               " commands; the file requires " +
                               llvm::Twine(*def->commands));
    if (!def->commands && builtin && out.totalCommands() != *builtin)
      llvm::errs() << "warning: row-op template '" << getRowOpName(kind)
                   << "' from " << primitiveFilePath()
                   << " compiled to " << out.totalCommands()
                   << " commands; the built-in definition takes " << *builtin
                   << "\n";
  } else if (auto required = getRequiredCommands(kind)) {
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
  case RowOpKind::GreaterStep: return "row_greater";
  case RowOpKind::Select: return "row_select";
  case RowOpKind::RangeScan00: return "row_scan_00";
  case RowOpKind::RangeScan01: return "row_scan_01";
  case RowOpKind::RangeScan10: return "row_scan_10";
  case RowOpKind::RangeScan11: return "row_scan_11";
  case RowOpKind::ScanFold: return "row_scan_fold";
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
  //  * the range-scan step's 4 WAS a timing skeleton rather than a working
  //    sequence -- expandRangeScan's second per-bit command wrote C_0 into the
  //    (!DCC1, T1) group, driving the ">= lower" accumulator to 1 every
  //    iteration so that chain never advanced, and no step read the bounds at
  //    all. It is now the same five-command sequence as here, written in that
  //    player's row groups, so the 5 below is a reference again rather than a
  //    reminder.
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
  // expandMinMax runs as two passes rather than one fused step: a comparison
  // costing 3 a bit, then a select costing 10. Its 13n + 2 is those two plus
  // the seed and the park, which is how this pair is priced too.
  case RowOpKind::GreaterStep: return 3;
  case RowOpKind::Select: return 10;
  case RowOpKind::RangeScan00:
  case RowOpKind::RangeScan01:
  case RowOpKind::RangeScan10:
  case RowOpKind::RangeScan11:
    return 5;                             // expandRangeScan, per bit
  case RowOpKind::ScanFold:
    return 0;   // the simulator folds over its own row groups, not as a step
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
  if (entry.compiled)
    return &entry.tmpl;
  entry.compiled = true;

  // A file definition is a request to compile that network, so it takes the
  // place of a hand-written sequence as well as of the built-in network.
  bool fromFile = lookupFilePrimitive(kind) != nullptr;
  if (auto handWritten =
          fromFile ? std::nullopt : getHandWrittenTemplate(backend, kind)) {
    entry.tmpl = std::move(*handWritten);
    entry.tmpl.handWritten = true;
    for (const RowOpInstr &instr : entry.tmpl.instrs)
      (instr.isAp ? entry.tmpl.numAP : entry.tmpl.numAAP)++;
    entry.tmpl.simulated = simulate(kind, entry.tmpl);
    // A hand-written sequence has nothing but this check behind it, and every
    // kernel compiled from here on would carry the mistake. Refusing to run is
    // the only honest response.
    if (!entry.tmpl.simulated) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "hand-written row-op template '" << getRowOpName(kind)
         << "' does not compute the primitive on a model of the substrate; it "
            "must not be used";
      llvm::report_fatal_error(llvm::StringRef(msg));
    }
    return &entry.tmpl;
  }

  if (!compileTemplate(kind, entry.tmpl)) {
    entry.failed = true;
    return nullptr;
  }
  // lime has already proved this one equivalent to its MIG. Running it through
  // the model too is what earns the model the right to vouch for a sequence lime
  // did not compile, so a disagreement here is reported rather than acted on.
  entry.tmpl.simulated = simulate(kind, entry.tmpl);
  // lime proved the sequence equivalent to the file's network; the model is
  // what says whether that network is the primitive. One that is not would be
  // priced and emitted as though it were, so it is refused outright.
  if (fromFile && !entry.tmpl.simulated)
    llvm::report_fatal_error(llvm::Twine("row-op template '") +
                             getRowOpName(kind) + "' from " +
                             primitiveFilePath() +
                             " does not compute " + getRowOpName(kind) +
                             " on the model of the substrate: the network in "
                             "the file is not that primitive");
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
  pad(os, "cmds", 7, false);
  pad(os, "AAP", 7, false);
  pad(os, "AP", 7, false);
  pad(os, "ref", 7, false);
  pad(os, "delta", 8, false);
  pad(os, "source", 9, false);
  os << "  proof\n";
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
    pad(os, delta, 8, false);
    pad(os, tmpl->handWritten ? "hand" : tmpl->fromFile ? "file" : "lime", 9,
        false);
    // What actually stands behind the sequence: lime's SAT equivalence check, the
    // exhaustive run on the substrate model, or both.
    std::string proof;
    if (tmpl->validated)
      proof = tmpl->simulated ? "lime+sim" : "lime (sim DISAGREES)";
    else
      proof = tmpl->simulated ? "sim" : "NONE";
    os << "  " << proof << "\n";
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
  // Shift-add over the bit-sliced operands: one row AND per partial-product
  // bit, and one full adder per accumulation, its carry coming from a
  // partial-product row rather than a resident cell.
  //
  // Every bit of the multiplier meets every bit of the multiplicand once, so
  // there are lhs * rhs partial products; the first multiplier bit only
  // initialises the accumulator, leaving lhs * (rhs - 1) additions. One AAP
  // ahead of them seeds the carry the accumulation starts from.
  //
  // This used to charge lhs * (rhs + 1) ANDs and lhs * rhs - 1 additions,
  // taken from execute_mul in 94_simdram_schedule_runner.c. That sequence runs
  // one accumulation round more than a shift-add needs -- it passes over the
  // low multiplier bit twice -- and drops one partial product unaccumulated in
  // its last round. The row-level expansion in PatternApply does neither, so it
  // was being priced for work it does not do: 5.9% high at 16 bits, 11.1% at 8,
  // and more the narrower the operands.
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::And, lhsWidth * rhsWidth});
  recipe.steps.push_back(
      {RowOpKind::FullAddLoaded, lhsWidth * (rhsWidth - 1)});
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
  recipe.steps.push_back({RowOpKind::ScanFold, 1});
  // two commands seed the chains, one per cell
  recipe.fixedAAP = 2;
  return recipe;
}

std::optional<RowOpRecipe> mlir::bits::getRecipeForMinMax(int64_t width) {
  if (width <= 0)
    return std::nullopt;
  // Compare, then select -- the same two passes the simulator's expandMinMax
  // runs, and the reason this is not built on the MinMax template: that one
  // emits BOTH the min and the max from a verdict it is handed, where a
  // `bits.min` is one output and has to compute the verdict itself.
  RowOpRecipe recipe;
  recipe.steps.push_back({RowOpKind::GreaterStep, width});
  recipe.steps.push_back({RowOpKind::Select, width});
  // one command seeds the verdict, one moves it into the cell the select reads
  // it and its complement from
  recipe.fixedAAP = 2;
  return recipe;
}
