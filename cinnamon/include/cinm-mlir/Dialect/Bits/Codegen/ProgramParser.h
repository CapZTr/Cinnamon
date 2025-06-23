#pragma once

#include <llvm/Support/LogicalResult.h>
#include <optional>
#include <string>
#include <variant>
#include <vector>


enum class BitwiseOperandType { T, DCC };
struct BitwiseOperand {
  BitwiseOperandType type;
  int index;
  bool inverted;
};

enum class AddressType { In, Out, Spill, Const, Bitwise };
struct Address {
  AddressType type;
  std::variant<int, bool, std::vector<BitwiseOperand>> data;
};

struct Instruction {
  enum class Type { AAP, AP } type;
  Address operand0;
  std::optional<Address> operand1;
};

class ProgramParser {
public:
  explicit ProgramParser(std::string input);

  llvm::LogicalResult parse();

  const std::vector<Instruction> &getProgram() const { return program; }

private:
  std::vector<Instruction> program;
  std::string input;
  size_t position = 0;
  size_t currentLine = 0;

  void skipWhitespace();
  char peek() const;
  void advance();
  int parseNumber();
  Address parseAddress();
  BitwiseOperand parseBitwiseOperand();
};
