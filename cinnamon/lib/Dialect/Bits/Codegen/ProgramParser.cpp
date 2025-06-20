#include "cinm-mlir/Dialect/Bits/Codegen/ProgramParser.h"

#include <cassert>
#include <cctype>
#include <llvm/Support/LogicalResult.h>
#include <utility>
#include <vector>


ProgramParser::ProgramParser(std::string input) : input(input) {}

llvm::LogicalResult ProgramParser::parse() {
  while (position < input.size()) {
    skipWhitespace();
    if (position >= input.size()) break;

    std::string instr;
    while (std::isalpha(peek())) {
      instr += peek();
      advance();
    }

    assert(!instr.empty());

    Instruction instruction;

    if (instr == "AAP") {
      instruction.type = Instruction::Type::AAP;
      skipWhitespace();
      instruction.operand0 = parseAddress();
      skipWhitespace();
      instruction.operand1 = parseAddress();
    } else {
      assert(instr == "AP");
      skipWhitespace();
      instruction.operand0 = parseAddress();
    }

    program.push_back(std::move(instruction));

    while (position < input.size() && input[position] != '\n') {
      advance();
    }
    if (peek() == '\n') advance();

  }

  return llvm::success();
}

void ProgramParser::skipWhitespace() {
  while (position < input.size() && std::isspace(input[position])) {
    if (input[position] == '\n') currentLine++;
    position++;
  }
}

char ProgramParser::peek() const {
  return position < input.size() ? input[position] : '\0';
}

void ProgramParser::advance() {
  if (position < input.size()) {
    if (input[position] == '\n') currentLine++;
    position++;
  }
}

llvm::LogicalResult ProgramParser::expect(char c) {
  if (peek() != c) return llvm::failure();
  advance();
  return llvm::success();
}

int ProgramParser::parseNumber() {
  int val = 0;
  while (std::isdigit(peek())) {
    val = val * 10 + (peek() - '0');
    advance();
  }
  return val;
}

Address ProgramParser::parseAddress() {
  const char c = peek();
  Address addr;

  if (c == 'I' || c == 'O' || c == 'S') {
    advance();
    int index = parseNumber();

    addr.type = (c == 'I') ? AddressType::In :
        (c == 'O') ? AddressType::Out : AddressType::Spill;
    addr.data = index;
  } else if (c == 'C') {
    advance();
    addr.type = AddressType::Const;
    if (peek() == '0') {
      addr.data = false;
    } else {
      assert(peek() == '1');
      addr.data = true;
    }
    advance();
  } else if (c == '~') {
    advance();
    BitwiseOperand operand = parseBitwiseOperand();
    operand.inverted = true;
    addr.type = AddressType::Bitwise;
    addr.data = std::vector<BitwiseOperand>{operand};
  } else if (c == '[') {
    advance();
    std::vector<BitwiseOperand> operands;

    while (true) {
      skipWhitespace();
      if (peek() == ']') {
        advance();
        break;
      }

      bool inverted = false;
      if (peek() == '~') {
        inverted = true;
        advance();
      }

      BitwiseOperand operand = parseBitwiseOperand();
      operand.inverted = inverted;
      operands.push_back(operand);

      skipWhitespace();
      if (peek() == ',') {
        advance();
      } else {
        assert(peek() == ']');
      }
    }

    addr.type = AddressType::Bitwise;
    addr.data = operands;
  } else {
    BitwiseOperand operand = parseBitwiseOperand();
    addr.type = AddressType::Bitwise;
    addr.data = std::vector<BitwiseOperand>{operand};
  }

  return addr;
}

BitwiseOperand ProgramParser::parseBitwiseOperand() {
  BitwiseOperand operand;

  if (peek() == 'T') {
    operand.type = BitwiseOperandType::T;
    advance();
    operand.index = parseNumber();
  } else {
    assert(input.substr(position, 3) == "DCC");
    position += 3;
    operand.type = BitwiseOperandType::DCC;
    operand.index = parseNumber();
  }

  return operand;
}
