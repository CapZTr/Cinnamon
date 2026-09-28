//===- PrimitiveFile.cpp - row-op primitives defined in an MLIR file ------===//
//
// A file of primitive definitions looks like this -- the full adder, written
// as the built-in one is:
//
//   func.func @row_add(%a: !bits.row<1>, %b: !bits.row<1>,
//                      %cin: !bits.row<1> {pud.resident = "DCC1"})
//       -> (!bits.row<1>, !bits.row<1> {pud.resident = "DCC1"}) {
//     %cout  = bits.maj %a, %b, %cin : ...
//     %nb    = bits.not %b : ...
//     %m     = bits.maj %a, %nb, %cin : ...
//     %ncout = bits.not %cout : ...
//     %sum   = bits.maj %b, %ncout, %m : ...
//     return %sum, %cout : !bits.row<1>, !bits.row<1>
//   }
//
// Functions are named after the primitive they replace, as rowop-sanity prints
// them. Arguments are the network's inputs and results its outputs, in the
// order the built-in definition has them. The body is a majority network:
// bits.maj, bits.not, and the constant rows bits.create_all_zero_row and
// bits.create_all_one_row (with the arith.constant their width takes).
//
//   pud.resident = "DCC1"      on an argument or result: keep it in that cell
//                              (DCC0, DCC1, T0..T3) -- a loop-carried value
//   pud.preoptimize = false    on the function: compile the network as written
//   pud.commands = 8           on the function: require that many commands
//
//===----------------------------------------------------------------------===//

#include "PrimitiveFile.h"

#include "cinm-mlir/Dialect/Bits/IR/BitsDialect.h"
#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include <map>
#include <string>

using namespace mlir;
using namespace mlir::bits;
using MIG = mockturtle::mig_network;

namespace {

llvm::cl::opt<std::string> clPrimitives(
    "pud-primitives",
    llvm::cl::desc(
        "MLIR file of row-op primitive networks to compile instead of the "
        "built-in ones: one func.func per primitive, named as rowop-sanity "
        "prints it, whose body is bits.maj / bits.not / constant rows. "
        "`pud.resident = \"DCC1\"` on an argument or result pins it to that "
        "cell; `pud.preoptimize = false` compiles the network as written; "
        "`pud.commands = N` requires N commands. See PrimitiveFile.cpp."),
    llvm::cl::value_desc("file"), llvm::cl::init(""));

llvm::cl::opt<unsigned> clApproxAddLowBits(
    "pud-approx-add-low-bits",
    llvm::cl::desc(
        "Compute the lowest N bits of every plain addition with the "
        "approximate cell (the row_add_approx primitive: a | b unless "
        "--pud-primitives redefines it) instead of the full adder. No carry "
        "ripples through them; the exact part above starts with carry 0. "
        "Reductions and multiplications stay exact. Pass it to every cinm-opt "
        "run of a kernel: the mapping prices it and the expansion emits it."),
    llvm::cl::value_desc("N"), llvm::cl::init(0));

[[noreturn]] void fail(const llvm::Twine &msg) {
  llvm::report_fatal_error(llvm::Twine("--pud-primitives=") + clPrimitives +
                           ": " + msg);
}

PrimitiveHint cell(const std::string &where, unsigned index,
                   llvm::StringRef name) {
  uint32_t n;
  llvm::StringRef rest = name;
  if (rest.consume_front("DCC") && !rest.getAsInteger(10, n) && n < 2)
    return {index, kFileDccCellType, n};
  rest = name;
  if (rest.consume_front("T") && !rest.getAsInteger(10, n) && n < 4)
    return {index, kFileComputeCellType, n};
  fail(where + ": pud.resident = \"" + name +
       "\" is not a cell; the cells are DCC0, DCC1 and T0..T3");
}

PrimitiveDefinition read(func::FuncOp func) {
  std::string where = ("@" + func.getSymName()).str();
  PrimitiveDefinition def;
  MIG &mig = def.mig;
  llvm::DenseMap<Value, MIG::signal> signals;
  for (BlockArgument arg : func.getArguments())
    signals[arg] = mig.create_pi();

  auto signal = [&](Value v) {
    auto it = signals.find(v);
    if (it == signals.end())
      fail(where + ": uses a value its network does not define");
    return it->second;
  };

  if (!func.getBody().hasOneBlock())
    fail(where + ": a network is one block, with no control flow");
  for (Operation &op : func.getBody().front()) {
    if (auto maj = dyn_cast<MajOp>(op)) {
      signals[maj.getResult()] = mig.create_maj(
          signal(maj.getLhs()), signal(maj.getRhs()), signal(maj.getThird()));
    } else if (auto inv = dyn_cast<NotOp>(op)) {
      signals[inv.getResult()] = !signal(inv.getInput());
    } else if (isa<CreateAllZeroRowOp>(op)) {
      signals[op.getResult(0)] = mig.get_constant(false);
    } else if (isa<CreateAllOneRowOp>(op)) {
      signals[op.getResult(0)] = mig.get_constant(true);
    } else if (isa<arith::ConstantOp>(op)) {
      // the width operand of a constant row; not part of the network
    } else if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      for (Value v : ret.getOperands())
        mig.create_po(signal(v));
    } else {
      fail(where + ": '" + op.getName().getStringRef() +
           "' is not part of a majority network; use bits.maj, bits.not, "
           "bits.create_all_zero_row and bits.create_all_one_row");
    }
  }

  for (unsigned i = 0; i < func.getNumArguments(); ++i)
    if (auto attr = func.getArgAttrOfType<StringAttr>(i, "pud.resident"))
      def.inputs.push_back(cell(where, i, attr.getValue()));
  for (unsigned i = 0; i < func.getNumResults(); ++i)
    if (auto attr = func.getResultAttrOfType<StringAttr>(i, "pud.resident"))
      def.outputs.push_back(cell(where, i, attr.getValue()));
  if (auto attr = func->getAttrOfType<BoolAttr>("pud.preoptimize"))
    def.preoptimize = attr.getValue();
  if (auto attr = func->getAttrOfType<IntegerAttr>("pud.commands"))
    def.commands = static_cast<unsigned>(attr.getInt());
  return def;
}

/// Every definition in the file, by primitive; read once per process, so that
/// the mapping pass that prices a primitive and the conversion that emits it
/// see the same network.
const std::map<unsigned, PrimitiveDefinition> &definitions() {
  static const std::map<unsigned, PrimitiveDefinition> defs = [] {
    std::map<unsigned, PrimitiveDefinition> out;
    if (clPrimitives.empty())
      return out;

    DialectRegistry registry;
    registry.insert<func::FuncDialect, arith::ArithDialect, BitsDialect>();
    MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();
    OwningOpRef<ModuleOp> module =
        parseSourceFile<ModuleOp>(clPrimitives.getValue(), &ctx);
    if (!module)
      fail("cannot read the file (the parser said why, above)");

    for (auto func : module->getOps<func::FuncOp>()) {
      std::optional<RowOpKind> kind;
      for (unsigned k = 0; k < static_cast<unsigned>(RowOpKind::NUM_KINDS); ++k)
        if (getRowOpName(static_cast<RowOpKind>(k)) == func.getSymName())
          kind = static_cast<RowOpKind>(k);
      if (!kind)
        fail("@" + func.getSymName() +
             " is not a primitive; the names are those rowop-sanity prints");
      out[static_cast<unsigned>(*kind)] = read(func);
    }
    return out;
  }();
  return defs;
}

} // namespace

const PrimitiveDefinition *mlir::bits::lookupFilePrimitive(RowOpKind kind) {
  const auto &defs = definitions();
  auto it = defs.find(static_cast<unsigned>(kind));
  return it == defs.end() ? nullptr : &it->second;
}

llvm::StringRef mlir::bits::primitiveFilePath() { return clPrimitives; }

void mlir::bits::setPrimitiveFilePath(llvm::StringRef path) {
  clPrimitives = path.str();
}

unsigned mlir::bits::getApproxAddLowBits() { return clApproxAddLowBits; }
