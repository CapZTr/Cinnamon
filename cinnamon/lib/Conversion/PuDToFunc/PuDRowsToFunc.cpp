//===- PuDRowsToFunc.cpp - PuD row ops onto a target's row-op interface ---===//
//
// Lowers the PuD row ops onto what a program can issue, leaving the loop
// structure alone so the rest of the pipeline can take it to LLVM IR normally.
//
// A row op becomes a call to the stub that issues it:
//
//   void rowop_ap (void *dst)
//   void rowop_aap(void *dst, void *src)
//
// An address does not become a call. The dialect names a row by where it is --
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
// Only bank and row are realised. A rank or subarray coordinate that is not
// zero has nowhere to go in the table, so it is refused here rather than
// silently aliasing another row; the dialect goes on carrying all five.
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
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
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

/// Declares a callee once per module.
func::FuncOp getOrDeclare(ModuleOp module, StringRef name, FunctionType type) {
  if (auto existing = module.lookupSymbol<func::FuncOp>(name))
    return existing;
  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
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

/// Refuses a coordinate whose unrealised dimensions are not zero.
LogicalResult requireUnusedDimensionsZero(Operation *op, Value rank,
                                          Value subarray) {
  if (!matchPattern(rank, m_Zero()))
    return op->emitOpError()
           << "names rank " << rank << ", but rows are only laid out per bank";
  if (!matchPattern(subarray, m_Zero()))
    return op->emitOpError() << "names subarray " << subarray
                             << ", but rows are only laid out per bank";
  return success();
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
    if (failed(requireUnusedDimensionsZero(op, adaptor.getRankID(),
                                           adaptor.getSubarrayID())))
      return failure();

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
    if (failed(requireUnusedDimensionsZero(op, adaptor.getRankID(),
                                           adaptor.getSubarrayID())))
      return failure();

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

struct AAPLowering : public OpConversionPattern<AAPOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AAPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto type = rewriter.getFunctionType({ptrType, ptrType}, {});
    func::FuncOp callee = getOrDeclare(module, "rowop_aap", type);
    // the stub takes (dst, src); the op names them the other way round
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, callee, ValueRange{adaptor.getDstAddr(), adaptor.getSrcAddr()});
    return success();
  }
};

struct APLowering : public OpConversionPattern<APOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(APOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto ptrType = LLVM::LLVMPointerType::get(op.getContext());
    auto type = rewriter.getFunctionType({ptrType}, {});
    func::FuncOp callee = getOrDeclare(module, "rowop_ap", type);
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee,
                                              ValueRange{adaptor.getAddr()});
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
    patterns.add<AAPLowering, APLowering>(converter, ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  RowSpace space;
};

} // namespace

std::unique_ptr<Pass> mlir::pud::createConvertPuDRowsToFuncPass() {
  return std::make_unique<ConvertPuDRowsToFunc>();
}
