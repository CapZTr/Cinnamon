#pragma once

#include <llvm/Support/LogicalResult.h>

#include <iostream>
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
  std::string str_repr;
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

  void printProgram() const {
    std::cout << "\n========== Program ==========\n";
    for (const auto &inst : program) {
      std::string s = inst.type == Instruction::Type::AP ? "AP  " : "AAP ";
      s += inst.operand0.str_repr;
      if (inst.operand1.has_value()) {
        s += " ";
        s += inst.operand1->str_repr;
      }
      std::cout << s << '\n';
    }
    std::cout << "=============================\n\n";
  }

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
  void normalizeBitwiseAddress(Address &addr);
};
