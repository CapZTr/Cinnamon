#include "cinm-mlir/Dialect/PuD/IR/PuDAttributes.h"

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <optional>

namespace mlir {
namespace pud {

StringRef stringifyRowGroup(RowGroup group) {
  switch (group) {
    case RowGroup::B: return "B";
    case RowGroup::C: return "C";
    case RowGroup::D: return "D";
  }
  llvm_unreachable("unknown RowGroup enum value");
}

std::optional<RowGroup> symbolizeRowGroup(StringRef str) {
  return StringSwitch<std::optional<RowGroup>>(str)
      .Case("B", RowGroup::B)
      .Case("C", RowGroup::C)
      .Case("D", RowGroup::D)
      .Default(std::nullopt);
}

} // namespace pud
} // namespace mlir
