//===- PuDRowsToFunc.cpp - PuD row ops onto a target's row-op interface ---===//
//
// Lowers the PuD row ops onto what a program can issue, leaving the loop
// structure alone so the rest of the pipeline can take it to LLVM IR normally.
//
// A row op becomes the target's instruction for it, emitted in place.
//
// An address does not become a call either, and it is not a lookup. The dialect
// names a row by where it is -- channel, rank, bank, subarray, row -- and the
// runtime that owns the row space lays it out at a fixed stride, so a
// coordinate is arithmetic:
//
//   pud_arena + (control rows + row) * (row stride * banks) + bank * row stride
//
// with the rows the substrate reserves at the front, in the order the dialect
// enumerates them.
//
// It used to be a load out of a table the runtime published, one entry per
// (row, bank). That table is as large as the kernel's row space -- 63 kB for
// popcount -- and a kernel walks all of it every wave, so it does not stay in a
// 32 kB L1, and every miss is a DRAM read that stops the row-op stream while
// the controller turns the bus around. Arithmetic has no such table: the bank
// loop becomes a pointer walking by the row stride, and the row coordinate
// folds into the loop that produced it.
//
// Rank and subarray are carried through the dialect but do not yet select
// anything: rows are laid out per bank, so there is nothing for those
// coordinates to pick between. Distributing rows over them is a later change,
// and it is this arithmetic that will gain the terms when it happens.
//
//===----------------------------------------------------------------------===//

#include "cinm-mlir/Conversion/PuDToFunc/PuDToFunc.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDDialect.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDOps.h"
#include "cinm-mlir/Dialect/PuD/IR/PuDTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

using namespace mlir;
using namespace mlir::pud;

#define GEN_PASS_CLASSES
#include "cinm-mlir/Conversion/PuDPasses.h.inc"

namespace {

/// The base of the row space, as the runtime publishes it.
constexpr StringLiteral kArenaSymbol = "pud_arena";

/// The shape of the row space, as the conversion to PuD recorded it.
///
/// Both the table layout and the flattening of a bank coordinate come from
/// here, so that a bank index means the same thing to this pass as it did to
/// the mapping that chose it.
struct RowSpace {
  int64_t banksPerChannel = 0;
  int64_t totalBanks = 0;
  int64_t dataRowsPerBank = 0;
  int64_t rowStride = 0;

  /// Bytes from one row of the space to the next.
  ///
  /// A row exists in every bank of the channel, not only in the banks this
  /// kernel was given: what turns an address back into a bank and a row is the
  /// controller's decode, and that counts in banks per channel. Laying the
  /// rows out any denser than that puts the second row of bank 0 in some other
  /// bank -- at eight banks of a thirty-two-bank channel, in bank eight -- and
  /// every row op after the first names rows the kernel never meant. The banks
  /// the kernel does not use are a hole in the space, which costs address
  /// range and nothing else.
  int64_t period() const { return rowStride * banksPerChannel; }
  /// Rows the substrate reserves, which sit at the front of the space.
  static int64_t controlRows() { return pud::controlRowCount(); }
};

FailureOr<RowSpace> readRowSpace(ModuleOp module) {
  RowSpace space;
  auto get = [&](StringRef name, int64_t &into) -> bool {
    auto attr = module->getAttrOfType<IntegerAttr>(name);
    if (!attr)
      return false;
    into = attr.getInt();
    return true;
  };
  if (!get("pud.banks_per_channel", space.banksPerChannel) ||
      !get("pud.total_banks", space.totalBanks) ||
      !get("pud.data_rows_per_bank", space.dataRowsPerBank) ||
      !get("pud.row_stride", space.rowStride)) {
    module.emitError() << "no row space on the module; run "
                          "convert-bits-rows-to-pud first";
    return failure();
  }
  if (space.banksPerChannel <= 0 || space.totalBanks <= 0) {
    module.emitError() << "the row space spans no banks";
    return failure();
  }
  if (space.rowStride <= 0) {
    module.emitError() << "a row has to be some bytes wide";
    return failure();
  }
  return space;
}

/// Emits the target's instruction for a row op.
///
/// A row op is one instruction: a two-byte opcode with the operation as its
/// immediate, reading the rows it acts on from the argument registers. It is
/// emitted here rather than reached through a stub because at the rate row ops
/// are issued a call is not small next to what it wraps -- it adds a push and a
/// pop of the return address to every one of them, and the push is an ordinary
/// store, which orders against every other store the program makes.
///
/// It has side effects and no result: what it leaves behind is in DRAM rows,
/// which nothing here can name, so nothing may reorder or drop it. That is what
/// keeps the emitted order the order the mapping decided on.
void emitRowOp(OpBuilder &builder, Location loc, unsigned opcode,
               ValueRange rows) {
  // The rows go in the argument registers, in the order the instruction reads
  // them; "memory" is what the row op leaves its result in.
  static constexpr StringLiteral kRegisters[] = {"{rdi}", "{rsi}", "{rdx}"};
  assert(rows.size() <= std::size(kRegisters) && "more rows than the "
                                                 "instruction takes");
  std::string constraints;
  for (size_t i = 0; i < rows.size(); ++i)
    constraints += std::string(kRegisters[i]) + ",";
  constraints += "~{memory}";

  std::string text = llvm::formatv(".byte 0x0F, 0x04\n.word {0}", opcode);
  builder.create<LLVM::InlineAsmOp>(
      loc, /*res=*/Type(), rows, text, constraints, /*has_side_effects=*/true,
      /*is_align_stack=*/false, /*asm_dialect=*/LLVM::AsmDialectAttr(),
      /*operand_attrs=*/ArrayAttr());
}

/// Declares the runtime's arena pointer once per module.
///
/// It is dso_local: the runtime that defines it is linked into the same program
/// as the kernel that reads it -- it is the runtime for this kernel -- so the
/// address is known at link time and an indirection through a global offset
/// table would be one more load in the row-op path for nothing.
void declareArena(ModuleOp module) {
  if (module.lookupSymbol<LLVM::GlobalOp>(kArenaSymbol))
    return;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  builder.create<LLVM::GlobalOp>(
      module.getLoc(), LLVM::LLVMPointerType::get(module.getContext()),
      /*isConstant=*/false, LLVM::Linkage::External, kArenaSymbol,
      /*value=*/Attribute(), /*alignment=*/0, /*addrSpace=*/0,
      /*dsoLocal=*/true);
}

/// The bank a coordinate names, as one index over the whole memory -- the same
/// flattening the mapping numbered banks with.
Value flattenBank(OpBuilder &builder, Location loc, Value channel, Value bank,
                  int64_t banksPerChannel) {
  Value perChannel =
      builder.create<arith::ConstantIntOp>(loc, banksPerChannel, 64);
  Value inChannels = builder.create<arith::MulIOp>(loc, channel, perChannel);
  return builder.create<arith::AddIOp>(loc, inChannels, bank);
}

/// The base of the row space, loaded once per function.
///
/// Once, because a row op clobbers memory as far as the compiler is concerned
/// -- that is what keeps the schedule in order -- so a load written out per
/// address could never be hoisted out from between two of them, and a kernel
/// would spend an instruction and a cache access per operand re-reading a
/// pointer that never changes.
class ArenaBase {
public:
  Value get(OpBuilder &builder, func::FuncOp func) {
    auto it = bases.find(func.getOperation());
    if (it != bases.end())
      return it->second;
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&func.getBody().front());
    Location loc = func.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    Value arena = builder.create<LLVM::AddressOfOp>(loc, ptrType, kArenaSymbol);
    Value base = builder.create<LLVM::LoadOp>(loc, ptrType, arena);
    bases[func.getOperation()] = base;
    return base;
  }

private:
  DenseMap<Operation *, Value> bases;
};

/// Where a row of the space is: its index times the period, plus the bank.
///
/// `rowInSpace` counts from the front of the space, where the substrate's
/// reserved rows are, so a data row is offset past them.
Value rowAddress(OpBuilder &builder, Location loc, const RowSpace &space,
                 Value base, Value rowInSpace, Value bank) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Value period = builder.create<arith::ConstantIntOp>(loc, space.period(), 64);
  Value stride =
      builder.create<arith::ConstantIntOp>(loc, space.rowStride, 64);
  Value offset = builder.create<arith::AddIOp>(
      loc, builder.create<arith::MulIOp>(loc, rowInSpace, period),
      builder.create<arith::MulIOp>(loc, bank, stride));
  return builder.create<LLVM::GEPOp>(loc, ptrType, builder.getI8Type(), base,
                                     ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(offset)},
                                     /*inbounds=*/true);
}

/// A data row: the row the coordinate names, in the bank it names.
struct GetRowLowering : public OpConversionPattern<GetRowOp> {
  GetRowLowering(const TypeConverter &converter, MLIRContext *ctx,
                 const RowSpace &space, ArenaBase &arena)
      : OpConversionPattern(converter, ctx), space(space), arena(arena) {}

  LogicalResult
  matchAndRewrite(GetRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bank = flattenBank(rewriter, loc, adaptor.getChannelID(),
                             adaptor.getBankID(), space.banksPerChannel);
    // Past the rows the substrate reserves, which are at the front.
    Value first = rewriter.create<arith::ConstantIntOp>(
        loc, RowSpace::controlRows(), 64);
    Value row = rewriter.create<arith::AddIOp>(loc, first, adaptor.getRowID());
    Value base = arena.get(rewriter, op->getParentOfType<func::FuncOp>());
    rewriter.replaceOp(op, rowAddress(rewriter, loc, space, base, row, bank));
    return success();
  }

  const RowSpace &space;
  ArenaBase &arena;
};

/// A control row: named rather than numbered, so the name is the row index.
struct GetCtrlRowLowering : public OpConversionPattern<GetCtrlRowOp> {
  GetCtrlRowLowering(const TypeConverter &converter, MLIRContext *ctx,
                     const RowSpace &space, ArenaBase &arena)
      : OpConversionPattern(converter, ctx), space(space), arena(arena) {}

  LogicalResult
  matchAndRewrite(GetCtrlRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bank = flattenBank(rewriter, loc, adaptor.getChannelID(),
                             adaptor.getBankID(), space.banksPerChannel);
    Value row = rewriter.create<arith::ConstantIntOp>(
        loc, static_cast<int64_t>(op.getCtrlRow()), 64);
    Value base = arena.get(rewriter, op->getParentOfType<func::FuncOp>());
    rewriter.replaceOp(op, rowAddress(rewriter, loc, space, base, row, bank));
    return success();
  }

  const RowSpace &space;
  ArenaBase &arena;
};

/// Which instruction each row op is. The encodings are the target's; the
/// operations they name are the substrate's.
enum RowOpcode : unsigned {
  kAP = 0x60,
  kAAP = 0x61,
  kRowCopy = 0x62,
};

struct AAPLowering : public OpConversionPattern<AAPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // the instruction takes (dst, src); the op names them the other way round
    emitRowOp(rewriter, op.getLoc(), kAAP,
              ValueRange{adaptor.getDstAddr(), adaptor.getSrcAddr()});
    rewriter.eraseOp(op);
    return success();
  }
};

/// A copy is its own instruction rather than an AAP, because its two rows can be
/// in different banks: an AAP activates one bank twice, and the substrate has to
/// know that this one does not.
struct RowCopyLowering : public OpConversionPattern<RowCopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RowCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitRowOp(rewriter, op.getLoc(), kRowCopy,
              ValueRange{adaptor.getDstAddr(), adaptor.getSrcAddr()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct APLowering : public OpConversionPattern<APOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(APOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    emitRowOp(rewriter, op.getLoc(), kAP, ValueRange{adaptor.getAddr()});
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertPuDRowsToFunc
    : public ConvertPuDRowsToFuncBase<ConvertPuDRowsToFunc> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Nothing to resolve, and so nothing to resolve it against.
    bool hasRowOps = module
                         .walk([](Operation *op) {
                           return isa<PuDDialect>(op->getDialect())
                                      ? WalkResult::interrupt()
                                      : WalkResult::advance();
                         })
                         .wasInterrupted();
    if (!hasRowOps)
      return;

    FailureOr<RowSpace> rowSpace = readRowSpace(module);
    if (failed(rowSpace))
      return signalPassFailure();
    space = *rowSpace;

    declareArena(module);

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    // a row address is a pointer once the coordinate has been resolved
    converter.addConversion([ctx](RowType) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });

    ConversionTarget target(*ctx);
    target.addIllegalDialect<PuDDialect>();
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect, LLVM::LLVMDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<GetRowLowering, GetCtrlRowLowering>(converter, ctx, space,
                                                    arena);
    patterns.add<AAPLowering, APLowering, RowCopyLowering>(converter, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  RowSpace space;
  ArenaBase arena;
};

} // namespace

std::unique_ptr<Pass> mlir::pud::createConvertPuDRowsToFuncPass() {
  return std::make_unique<ConvertPuDRowsToFunc>();
}
