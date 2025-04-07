#ifndef CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H
#define CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H

#include "cinm-mlir/Dialect/Bits/IR/BitsOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <vector>

namespace mlir::bits {

struct BitplaneData {
  int64_t bitWidth;
  int64_t vectorLength;
};

struct AddData {
  BitplaneData lhs;
  BitplaneData rhs;
  BitplaneData result;
};

class BitsParser {
public:
  explicit BitsParser(ModuleOp module);

  void parse();

  const llvm::DenseMap<Value, BitplaneData> &getParsed() const { return parsed; }
  const llvm::DenseMap<Value, BitplaneData> &getInputs() const { return inputs; }
  const std::vector<AddData> &getAdds() const { return adds; }

private:
  ModuleOp module;
  llvm::DenseMap<Value, BitplaneData> parsed;
  llvm::DenseMap<Value, BitplaneData> inputs;
  std::vector<AddData> adds;

  void parseTranspose(TransposeOp op);
  void parseAdd(AddOp op);
  bool operandsParsed(Operation *op) const;
};

} // namespace mlir::bits

namespace llvm {
template <>
struct DenseMapInfo<mlir::bits::BitplaneData> {
  static inline mlir::bits::BitplaneData getEmptyKey() {
    return {~0LL, ~0LL};
  }

  static inline mlir::bits::BitplaneData getTombstoneKey() {
    return {~0LL - 1, ~0LL - 1};
  }

  static unsigned getHashValue(const mlir::bits::BitplaneData &val) {
    return hash_combine(val.bitWidth, val.vectorLength);
  }

  static bool isEqual(const mlir::bits::BitplaneData &lhs,
                      const mlir::bits::BitplaneData &rhs) {
    return lhs.bitWidth == rhs.bitWidth &&
           lhs.vectorLength == rhs.vectorLength;
  }
};
}

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H
