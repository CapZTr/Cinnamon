#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

namespace mlir::bits {

struct BitplaneData {
  int64_t bitWidth;
  int64_t vectorLength;
  Value slice;

  bool operator==(const BitplaneData &other) const {
    return slice == other.slice;
  }
};

struct BinaryOpData {
  BitplaneData lhs;
  BitplaneData rhs;
  BitplaneData result;

  bool operator==(const BinaryOpData &other) const {
    return lhs == other.lhs && rhs == other.rhs && result == other.result;
  }
};

class BitsParser {
public:
  explicit BitsParser(ModuleOp module);

  LogicalResult parse();

  const llvm::DenseMap<Value, BitplaneData> &getParsed() const { return parsed; }
  const llvm::DenseMap<Value, BitplaneData> &getInputs() const { return inputs; }
  const llvm::DenseMap<Value, BitplaneData> &getOutputs() const { return outputs; }
  const llvm::SmallVector<BinaryOpData, 8> &getBinaryOps() const { return binaryOps; }
  const llvm::SmallVector<BinaryOpData, 4> &getAdds() const { return adds; }
  const llvm::SmallVector<BinaryOpData, 4> &getSubs() const { return subs; }

private:
  ModuleOp module;
  llvm::DenseMap<Value, BitplaneData> parsed;
  llvm::DenseMap<Value, BitplaneData> inputs;
  llvm::DenseMap<Value, BitplaneData> outputs;
  llvm::SmallVector<BinaryOpData, 8> binaryOps;
  llvm::SmallVector<BinaryOpData, 4> adds;
  llvm::SmallVector<BinaryOpData, 4> subs;

  void parseTranspose(TransposeOp op);
  LogicalResult parseBinaryOp(Operation *op);
  bool operandsParsed(Operation *op) const;
};

} // namespace mlir::bits

namespace llvm {
template <>
struct DenseMapInfo<mlir::bits::BitplaneData> {
  static inline mlir::bits::BitplaneData getEmptyKey() {
    return {~0LL - 1, ~0LL - 1, mlir::Value::getFromOpaquePointer((LogicalResult*) - 1)};
  }

  static inline mlir::bits::BitplaneData getTombstoneKey() {
    return {~0LL - 2, ~0LL - 2, mlir::Value::getFromOpaquePointer((LogicalResult*) - 2)};
  }

  static unsigned getHashValue(const mlir::bits::BitplaneData &val) {
    return llvm::hash_value(val.slice.getAsOpaquePointer());
  }

  static bool isEqual(const mlir::bits::BitplaneData &lhs,
                      const mlir::bits::BitplaneData &rhs) {
    return lhs.slice == rhs.slice;
  }
};
}

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H
