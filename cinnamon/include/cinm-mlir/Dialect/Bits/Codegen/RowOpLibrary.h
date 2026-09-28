//===- RowOpLibrary.h - row-op primitives compiled by lime ------*- C++ -*-===//
//
// Single source of truth for how a bit-serial row operation is realised on a
// processing-using-DRAM substrate.
//
// Each row-op primitive is described once as a small majority-inverter graph,
// handed to lime (with SAT equivalence checking on), and cached as an
// instruction template together with its AAP/AP counts. Two consumers share
// that one answer:
//
//   * the ILP mapping, which needs the counts to price a slice-level op, and
//   * code generation, which needs the instruction sequence itself.
//
// Keeping both on the same templates is the point: an op's modelled cost and
// the code emitted for it cannot drift apart.
//
// A few primitives carry a hand-written sequence instead. lime's scheduler is
// not always shortest: for XNOR it schedules the complement of an XOR and pays
// two commands for the inversion, where a seven-command program exists. Where a
// shorter sequence is known, it is written out here and lime's is not used --
// but never on trust. Every template, compiled or written, is run against a
// model of the substrate over all input assignments and all initial states of
// the compute rows; agreement across the templates lime has already proved
// equivalent is what makes that model worth checking the hand-written ones with.
//
//===----------------------------------------------------------------------===//

#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_ROWOPLIBRARY_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_ROWOPLIBRARY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::bits {

//===----------------------------------------------------------------------===//
// Substrate
//===----------------------------------------------------------------------===//

enum class PudBackend {
  /// Ambit-style triple-row activation with dual-contact cells. The only
  /// backend lime has an architecture definition for today.
  Simdram,
  /// Single-command 3-row majority, no dual-contact cell. Priced from the
  /// legacy table until lime gains a PRADA definition.
  Prada,
  /// COTS DDR4 cross-subarray gates plus FracDRAM majority.
  Fcdram,
};

llvm::StringRef getBackendName(PudBackend backend);

//===----------------------------------------------------------------------===//
// Control rows
//===----------------------------------------------------------------------===//

/// The rows a subarray reserves for computation.
///
/// Four compute rows and two dual-contact cells, each presenting its value on
/// one row and its complement on another. The compound names are the row sets a
/// single activation drives: the pairs, and the triples whose activation leaves
/// the majority of the three in every one of them. Which sets exist is a
/// property of the substrate -- they are wired that way -- so they are named
/// rather than derived. Two constant rows, held at 0 and at 1, complete the set.
enum class ControlRow {
  T0,
  T1,
  T2,
  T3,
  DCC0,
  DCC0N,
  DCC1,
  DCC1N,
  DCC0N_T0,
  DCC1N_T1,
  T2_T3,
  T0_T3,
  T0_T1_T2,
  T0_T1_T3,
  DCC0_T1_T3,
  DCC1_T0_T2,
  C_0,
  C_1,

  NUM_CONTROL_ROWS,
};

llvm::StringRef getControlRowName(ControlRow row);

//===----------------------------------------------------------------------===//
// Instruction templates
//===----------------------------------------------------------------------===//

/// One operand of a templated row op: either a fixed control row of the
/// substrate, or a placeholder the caller binds to a data row.
struct RowOperand {
  enum class Kind {
    /// A row the subarray reserves for computation; `index` is a ControlRow.
    Control,
    /// The template's `index`-th data input.
    Input,
    /// The template's `index`-th data output.
    Output,
    /// A scratch data row the template needed internally.
    Temp,
  };

  Kind kind = Kind::Control;
  int index = 0;

  static RowOperand control(ControlRow row) {
    return {Kind::Control, static_cast<int>(row)};
  }
  ControlRow getControlRow() const { return static_cast<ControlRow>(index); }
  static RowOperand input(int i) { return {Kind::Input, i}; }
  static RowOperand output(int i) { return {Kind::Output, i}; }
  static RowOperand temp(int i) { return {Kind::Temp, i}; }

  bool isControl() const { return kind == Kind::Control; }
  std::string str() const;

  bool operator==(const RowOperand &o) const {
    return kind == o.kind && index == o.index;
  }
};

/// A single emitted row op.
///
/// An AP activates one row group and leaves the majority in every row of it. An
/// AAP additionally copies a source row into a destination group first, which is
/// how both operand staging and result write-back are expressed.
struct RowOpInstr {
  /// true for AP (activate the group named by `dst`, no source read).
  bool isAp = false;
  /// Source row for an AAP. Unused when `isAp`.
  RowOperand src;
  /// Destination row group; for an AP, the group being activated.
  RowOperand dst;

  std::string str() const;
};

/// A compiled row-op primitive.
struct RowOpTemplate {
  llvm::SmallVector<RowOpInstr, 16> instrs;

  /// Command counts, split the way the DRAM timing model charges them.
  unsigned numAAP = 0;
  unsigned numAP = 0;
  unsigned numInputs = 0;
  unsigned numOutputs = 0;

  /// Rows the caller must have loaded before the sequence runs, and where each
  /// result lands. A value appearing in both is loop-carried: it stays resident
  /// in a dual-contact cell across bit iterations, which is what makes a
  /// bit-serial carry cost nothing to keep.
  llvm::SmallVector<RowOperand, 4> inputBindings;
  llvm::SmallVector<RowOperand, 4> outputBindings;

  /// Whether lime proved the compiled program equivalent to the source MIG.
  /// Always false for a hand-written template: lime proves what it compiles, and
  /// a sequence it did not compile is outside what it can speak about.
  bool validated = false;

  /// Set when this sequence was written by hand rather than compiled, because
  /// lime's scheduler does not find the shortest program for this primitive. A
  /// hand-written template is only ever accepted when `simulated` holds.
  bool handWritten = false;

  /// Set when the network lime compiled came from the file named by
  /// --pud-primitives rather than from the built-in definition. Such a template
  /// is only ever accepted when `simulated` holds: the file says what network
  /// to compile, not what the primitive means.
  bool fromFile = false;

  /// Whether running the sequence on a model of the substrate reproduced the
  /// primitive's truth table over every input assignment. This is what a
  /// hand-written template stands on; it is also run over lime's templates,
  /// where agreeing with fifteen programs lime has already proved is what makes
  /// the model itself trustworthy.
  bool simulated = false;

  unsigned totalCommands() const { return numAAP + numAP; }
};

//===----------------------------------------------------------------------===//
// Row-op primitives
//===----------------------------------------------------------------------===//

/// The bit-serial primitives slice-level ops are built from. Each is one small
/// MIG; nothing here grows with an operand's bit width -- width shows up in the
/// recipe that repeats these, never inside a template.
enum class RowOpKind {
  /// out = a & b
  And,
  /// out = a | b
  Or,
  /// out = !a
  Not,
  /// out = maj(a, b, c)
  Maj3,
  /// out = a ^ b
  Xor,
  /// out = !(a ^ b) -- the BNN multiply
  Xnor,
  /// {sum, cout} = a + b + cin, with the carry loop-carried in a DCC. Costs one
  /// command less than FullAddLoaded because the carry never touches a data row.
  FullAdd,
  /// The same full adder with the carry-in read from, and the carry-out written
  /// to, caller-supplied data rows. This is what a shift-add multiplier needs:
  /// its carries come from partial-product rows, not from a resident cell.
  FullAddLoaded,
  /// {diff, bout} = a - b - bin, the adder with the subtrahend complemented
  FullSub,
  /// x & !sign -- the ReLU bit step
  Relu,
  /// {min, max} = compare-and-select given an "a greater" flag
  MinMax,
  /// One bit of the comparison a min or a max starts with: g' = maj(a, !b, g),
  /// the whole comparator being a single majority per bit. The verdict stays
  /// resident in a compute row, not a dual-contact cell -- it is read three
  /// times a bit and complemented none.
  GreaterStep,
  /// One bit of the select a min or a max ends with: out = g ? x : y, with the
  /// verdict resident in a dual-contact cell so that both it and its complement
  /// are readable without a command.
  Select,
  /// One BitWeaving range-scan bit step, advancing both comparison chains.
  /// The scan bounds are compile-time constants, so this comes in four
  /// variants; see getRangeScanKind.
  RangeScan00,
  RangeScan01,
  RangeScan10,
  RangeScan11,
  /// Combines the two range-scan chains into the match mask. Its own primitive
  /// because both operands are already resident in dual-contact cells, which a
  /// general three-input majority does not know and would pay a command to
  /// restate.
  ScanFold,
  /// One low bit of an approximate addition: out = f(a, b), no carry in or
  /// out. The function is the network's -- a | b built in, or whatever
  /// --pud-primitives defines -- so the network is also what it is checked
  /// against.
  ApproxAdd,

  NUM_KINDS,
};

llvm::StringRef getRowOpName(RowOpKind kind);

/// How many low bits of every plain addition the approximate cell computes
/// (--pud-approx-add-low-bits); 0 when additions are exact.
unsigned getApproxAddLowBits();

/// The range-scan step for a given (lower[j], upper[j]) bit pair.
RowOpKind getRangeScanKind(bool lowerBit, bool upperBit);

//===----------------------------------------------------------------------===//
// Library
//===----------------------------------------------------------------------===//

/// Command counts for one row-op primitive.
struct RowOpCost {
  unsigned numAAP = 0;
  unsigned numAP = 0;
  unsigned total() const { return numAAP + numAP; }
};

/// Compiles row-op primitives through lime and caches the result for the
/// lifetime of the process. Templates are compiled lazily: a run that never asks
/// for a primitive never pays for it.
class RowOpLibrary {
public:
  static RowOpLibrary &get();

  /// Returns the compiled template, or nullptr if this backend has no lime
  /// architecture definition yet (everything but SIMDRAM today).
  const RowOpTemplate *getTemplate(PudBackend backend, RowOpKind kind);

  /// Command counts for one instance of the primitive. Returns nullopt for
  /// backends lime cannot compile for; callers fall back to the legacy table.
  std::optional<RowOpCost> getCost(PudBackend backend, RowOpKind kind);

  /// Renders every compiled template with its counts, next to the hand-written
  /// expansion counts in rowop_trace_player.cc where one exists. Differences are
  /// reported, not hidden -- see the header comment on kReferenceCommands.
  std::string renderSanityTable(PudBackend backend);

private:
  RowOpLibrary() = default;
  RowOpLibrary(const RowOpLibrary &) = delete;

  struct Entry {
    RowOpTemplate tmpl;
    bool compiled = false;
    bool failed = false;
  };
  Entry &entryFor(PudBackend backend, RowOpKind kind);

  // [backend][kind]
  Entry cache[3][static_cast<unsigned>(RowOpKind::NUM_KINDS)];
};

/// Command count of the hand-written expansion in rowop_trace_player.cc, or 0
/// where the simulator has no sequence for the primitive. These are the numbers
/// the legacy cost table was built from; the sanity table prints them beside
/// lime's so any divergence is visible rather than silently absorbed.
unsigned getReferenceCommands(PudBackend backend, RowOpKind kind);

//===----------------------------------------------------------------------===//
// Recipes
//===----------------------------------------------------------------------===//

/// How many of each row-op primitive a slice-level op expands into, plus the
/// fixed commands that bracket the loop (carry initialisation, result
/// finalisation). Shared by the ILP cost model and the slice-to-row lowering so
/// the price of an op and the code emitted for it stay the same thing.
struct RowOpRecipe {
  struct Step {
    RowOpKind kind;
    /// How many times this primitive runs.
    int64_t count;
  };

  llvm::SmallVector<Step, 4> steps;
  /// AAP commands outside the row-op steps: seeding a carry, storing a
  /// carry-out, folding two chains into a mask, and so on.
  int64_t fixedAAP = 0;
  int64_t fixedAP = 0;

  /// Total commands under `backend`, or nullopt if any step is uncompilable
  /// there.
  std::optional<int64_t> totalCommands(PudBackend backend) const;
};

/// The recipe for a slice-level op. `lhsWidth` / `rhsWidth` are the operand bit
/// widths; ops that only care about one width ignore the second.
///
/// Returns nullopt for ops with no row-level expansion, which the caller prices
/// with its fallback.
std::optional<RowOpRecipe> getRecipeForAdd(int64_t width, bool widening);
/// A plain addition whose `lowBits` lowest bits are computed by the
/// approximate cell and the rest by the full adder, the carry starting at zero
/// where the exact part starts.
std::optional<RowOpRecipe> getRecipeForApproxAdd(int64_t width,
                                                 int64_t lowBits);
std::optional<RowOpRecipe> getRecipeForSub(int64_t width);
std::optional<RowOpRecipe> getRecipeForMul(int64_t lhsWidth, int64_t rhsWidth);
std::optional<RowOpRecipe> getRecipeForAnd(int64_t width, bool maskedZero);
std::optional<RowOpRecipe> getRecipeForOr(int64_t width);
std::optional<RowOpRecipe> getRecipeForXnor(int64_t width);
std::optional<RowOpRecipe> getRecipeForRangeScan(int64_t width);
std::optional<RowOpRecipe> getRecipeForMinMax(int64_t width);

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_ROWOPLIBRARY_H
