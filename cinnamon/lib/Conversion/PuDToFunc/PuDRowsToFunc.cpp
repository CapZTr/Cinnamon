//===- PuDRowsToFunc.cpp - PuD row ops onto a target's row-op interface ---===//
//
// Lowers the PuD row ops onto what a program can issue, leaving the loop
// structure alone so the rest of the pipeline can take it to LLVM IR normally.
//
// A row op becomes the target's instruction for it, emitted in place.
//
// An address does not become a call either. The dialect names a row by where it is --
// channel, rank, bank, subarray, row -- and the runtime that owns the row
// allocation publishes where each row it allocated actually landed, as tables
// indexed by that coordinate:
//
//   void *pud_ctrl_row_addr[<control rows>][<banks>]
//   void *pud_row_addr     [<rows per bank>][<banks>]
//
// so resolving a coordinate is a load, emitted here. Turning it into a call
// instead would hide it: the coordinate of a row op inside a loop is affine in
// the induction variables, and nothing downstream can see that through a call
// into another translation unit -- the redundant halves of the flattening stay,
// and the part that does not vary with the innermost loop cannot leave it.
//
// The two tables follow the coordinate. Control rows are named rather than
// numbered because the substrate reserves them: where they are is a property of
// it, not of what the allocation happened to hand out.
//
// The tables are indexed by bank and row alone. Rank and subarray are carried
// through the dialect but do not yet select anything: rows are laid out per
// bank, so there is nothing for those coordinates to pick between. Distributing
// rows over them is a later change, and it is these tables that will gain the
// dimensions when it happens.
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

/// Names of the row-address tables the runtime publishes.
constexpr StringLiteral kCtrlRowTable = "pud_ctrl_row_addr";
constexpr StringLiteral kDataRowTable = "pud_row_addr";

/// The shape of the row space, as the conversion to PuD recorded it.
///
/// Both the table layout and the flattening of a bank coordinate come from
/// here, so that a bank index means the same thing to this pass as it did to
/// the mapping that chose it.
struct RowSpace {
  int64_t banksPerChannel = 0;
  int64_t totalBanks = 0;
  int64_t dataRowsPerBank = 0;

  /// Type of a table with `rows` rows in every bank.
  LLVM::LLVMArrayType tableType(MLIRContext *ctx, int64_t rows) const {
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    return LLVM::LLVMArrayType::get(
        LLVM::LLVMArrayType::get(ptr, totalBanks), rows);
  }
  LLVM::LLVMArrayType ctrlTableType(MLIRContext *ctx) const {
    return tableType(ctx, getMaxEnumValForSimdramCtrlRow() + 1);
  }
  LLVM::LLVMArrayType dataTableType(MLIRContext *ctx) const {
    return tableType(ctx, dataRowsPerBank);
  }
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
      !get("pud.data_rows_per_bank", space.dataRowsPerBank)) {
    module.emitError() << "no row space on the module; run "
                          "convert-bits-rows-to-pud first";
    return failure();
  }
  if (space.banksPerChannel <= 0 || space.totalBanks <= 0) {
    module.emitError() << "the row space spans no banks";
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

/// Declares one of the runtime's tables once per module.
///
/// The table is dso_local: the runtime that defines it is linked into the same
/// program as the kernel that reads it -- it is the runtime for this kernel --
/// so the address is known at link time and an indirection through a global
/// offset table would be one more load in the row-op path for nothing.
void declareTable(ModuleOp module, StringRef name, Type type) {
  if (module.lookupSymbol<LLVM::GlobalOp>(name))
    return;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  builder.create<LLVM::GlobalOp>(module.getLoc(), type, /*isConstant=*/false,
                                 LLVM::Linkage::External, name,
                                 /*value=*/Attribute(), /*alignment=*/0,
                                 /*addrSpace=*/0, /*dsoLocal=*/true);
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

/// Reads a row's address out of one of the runtime's tables.
Value lookUpRow(OpBuilder &builder, Location loc, StringRef table,
                Type tableType, LLVM::GEPArg row, Value bank) {
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Value base = builder.create<LLVM::AddressOfOp>(loc, ptrType, table);
  SmallVector<LLVM::GEPArg, 3> index{LLVM::GEPArg(0), row, LLVM::GEPArg(bank)};
  Value slot = builder.create<LLVM::GEPOp>(loc, ptrType, tableType, base, index,
                                           /*inbounds=*/true);
  return builder.create<LLVM::LoadOp>(loc, ptrType, slot);
}

/// A data row: the row the coordinate names, in the bank it names.
struct GetRowLowering : public OpConversionPattern<GetRowOp> {
  GetRowLowering(const TypeConverter &converter, MLIRContext *ctx,
                 const RowSpace &space)
      : OpConversionPattern(converter, ctx), space(space) {}

  LogicalResult
  matchAndRewrite(GetRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bank = flattenBank(rewriter, loc, adaptor.getChannelID(),
                             adaptor.getBankID(), space.banksPerChannel);
    rewriter.replaceOp(
        op, lookUpRow(rewriter, loc, kDataRowTable,
                      space.dataTableType(op.getContext()), adaptor.getRowID(),
                      bank));
    return success();
  }

  const RowSpace &space;
};

/// A control row: named rather than numbered, so the name is the row index.
struct GetCtrlRowLowering : public OpConversionPattern<GetCtrlRowOp> {
  GetCtrlRowLowering(const TypeConverter &converter, MLIRContext *ctx,
                     const RowSpace &space)
      : OpConversionPattern(converter, ctx), space(space) {}

  LogicalResult
  matchAndRewrite(GetCtrlRowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value bank = flattenBank(rewriter, loc, adaptor.getChannelID(),
                             adaptor.getBankID(), space.banksPerChannel);
    rewriter.replaceOp(
        op, lookUpRow(rewriter, loc, kCtrlRowTable,
                      space.ctrlTableType(op.getContext()),
                      static_cast<int32_t>(op.getCtrlRow()), bank));
    return success();
  }

  const RowSpace &space;
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

    declareTable(module, kCtrlRowTable, space.ctrlTableType(ctx));
    declareTable(module, kDataRowTable, space.dataTableType(ctx));

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
    patterns.add<GetRowLowering, GetCtrlRowLowering>(converter, ctx, space);
    patterns.add<AAPLowering, APLowering, RowCopyLowering>(converter, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  RowSpace space;
};

} // namespace

std::unique_ptr<Pass> mlir::pud::createConvertPuDRowsToFuncPass() {
  return std::make_unique<ConvertPuDRowsToFunc>();
}
