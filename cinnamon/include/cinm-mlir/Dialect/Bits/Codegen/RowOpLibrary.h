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

/// Control-row slots, mirroring RowOpTracePlayer's slot indices and the rows
/// init_ambit() sets up in the guest runtime. Slots 0-17 are control rows;
/// data pools start at 18.
///
/// The group slots name the row sets a single activation drives. Slots 8-11 are
/// the dual-row activate (DRA) pairs and 12-15 the triple-row activate (TRA)
/// triples; both are taken from lime's SIMDRAM architecture definition, which
/// is what the templates below are compiled against.
///
/// NOTE: the DRA pairs already agree with the simulator, but its TRA triples are
/// currently Ambit's -- (T1,T2,T3), (DCC0,T1,T2), (DCC1,T0,T3) -- and three of
/// the four differ from SIMDRAM's. The simulator's slot table and init_ambit()
/// have to be switched over before generated code can run; that is a
/// replacement, not an addition, so slot numbering, the data pool base and the
/// trace format are all unaffected.
enum RowSlot : int {
  SLOT_T0 = 0,
  SLOT_T1,
  SLOT_T2,
  SLOT_T3,
  SLOT_DCC0,
  SLOT_DCC0N,
  SLOT_DCC1,
  SLOT_DCC1N,
  // dual-row activate pairs
  SLOT_DCC0N_T0,
  SLOT_DCC1N_T1,
  SLOT_T2_T3,
  SLOT_T0_T3,
  // triple-row activate groups (SIMDRAM)
  SLOT_T0_T1_T2,
  SLOT_T0_T1_T3,
  SLOT_DCC0_T1_T3,
  SLOT_DCC1_T0_T2,
  // constant rows
  SLOT_C_0,
  SLOT_C_1,

  SLOT_DATA_BASE = 18,
};

llvm::StringRef getSlotName(int slot);

//===----------------------------------------------------------------------===//
// Instruction templates
//===----------------------------------------------------------------------===//

/// One operand of a templated row op: either a fixed control row of the
/// substrate, or a placeholder the caller binds to a data row.
struct RowOperand {
  enum class Kind {
    /// A control row or row group; `index` is a RowSlot.
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

  static RowOperand control(int slot) { return {Kind::Control, slot}; }
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
  /// bit-serial carry free.
  llvm::SmallVector<RowOperand, 4> inputBindings;
  llvm::SmallVector<RowOperand, 4> outputBindings;

  /// Whether lime proved the compiled program equivalent to the source MIG.
  bool validated = false;

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
  /// One BitWeaving range-scan bit step, advancing both comparison chains.
  /// The scan bounds are compile-time constants, so this comes in four
  /// variants; see getRangeScanKind.
  RangeScan00,
  RangeScan01,
  RangeScan10,
  RangeScan11,

  NUM_KINDS,
};

llvm::StringRef getRowOpName(RowOpKind kind);

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
std::optional<RowOpRecipe> getRecipeForSub(int64_t width);
std::optional<RowOpRecipe> getRecipeForMul(int64_t lhsWidth, int64_t rhsWidth);
std::optional<RowOpRecipe> getRecipeForAnd(int64_t width, bool maskedZero);
std::optional<RowOpRecipe> getRecipeForOr(int64_t width);
std::optional<RowOpRecipe> getRecipeForXnor(int64_t width);
std::optional<RowOpRecipe> getRecipeForRangeScan(int64_t width);
std::optional<RowOpRecipe> getRecipeForMinMax(int64_t width);

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_ROWOPLIBRARY_H
