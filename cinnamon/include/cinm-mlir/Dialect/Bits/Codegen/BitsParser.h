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

  std::string toString() const {
    std::ostringstream oss;
    oss << "BitplaneData{bitWidth=" << bitWidth 
        << ", vectorLength=" << vectorLength << "}";
    return oss.str();
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

  const DenseMap<Value, BitplaneData> &getParsed() const { return parsed; }
  const DenseMap<Value, BitplaneData> &getInputs() const { return inputs; }
  const DenseMap<Value, BitplaneData> &getOutputs() const { return outputs; }
  const SmallVector<BinaryOpData, 8> &getBinaryOps() const { return binaryOps; }
  const SmallVector<BinaryOpData, 4> &getAdds() const { return adds; }

private:
  ModuleOp module;
  DenseMap<Value, BitplaneData> parsed;
  DenseMap<Value, BitplaneData> inputs;
  DenseMap<Value, BitplaneData> outputs;
  SmallVector<BinaryOpData, 8> binaryOps;
  SmallVector<BinaryOpData, 4> adds;

  void parseTranspose(TransposeOp op);
  void parseAssemble(AssembleOp op);
  LogicalResult parseBinaryOp(Operation *op);
  bool operandsParsed(Operation *op) const;
};

} // namespace mlir::bits

#endif // CINM_MLIR_DIALECT_BITS_CODEGEN_BITSPARSER_H
