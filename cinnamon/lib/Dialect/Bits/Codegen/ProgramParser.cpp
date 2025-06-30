#include "cinm-mlir/Dialect/Bits/Codegen/ProgramParser.h"

#include <cassert>
#include <cctype>
#include <format>
#include <iostream>
#include <llvm/Support/LogicalResult.h>
#include <utility>
#include <variant>
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
      instruction.type = Instruction::Type::AP;
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
    addr.str_repr = std::format("{}{}", c, index);
  } else if (c == 'C') {
    advance();
    addr.type = AddressType::Const;
    if (peek() == '0') {
      addr.data = false;
      addr.str_repr = std::format("{}{}", c, 0);
    } else {
      assert(peek() == '1');
      addr.data = true;
      addr.str_repr = std::format("{}{}", c, 1);
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

  if (addr.type == AddressType::Bitwise) normalizeBitwiseAddress(addr);

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

void ProgramParser::normalizeBitwiseAddress(Address &addr) {
  assert(addr.type == AddressType::Bitwise &&
      std::holds_alternative<std::vector<BitwiseOperand>>(addr.data));
  
  auto &data = std::get<std::vector<BitwiseOperand>>(addr.data);

  int index = -1;
  switch (data.size()) {

    case 1: {
      auto operand = data[0];
      if (operand.type == BitwiseOperandType::T) {
        assert(!operand.inverted);
        index = operand.index;
      } else {
        if (operand.inverted) {
          if (operand.index == 0) index = 5;
          else index = 7;
        } else {
          if (operand.index == 0) index = 4;
          else index = 6;
        }
      }
      break;
    }

    case 2: {
      auto operand0 = data[0];
      auto operand1 = data[1];
      if (operand0.type == BitwiseOperandType::T) {
        assert(!operand0.inverted && !operand1.inverted && operand1.index == 3);
        if (operand0.index == 2) index = 10;
        else {
          assert(operand0.index == 0);
          index = 11;
        }
      } else {
        assert(operand0.inverted &&
            operand1.type == BitwiseOperandType::T &&
            !operand1.inverted);
        if (operand0.index == 0) {
          assert(operand1.index == 0);
          index = 8;
        } else {
          assert(operand1.index == 1);
          index = 9;
        }
      }
      break;
    }

    case 3: {
      auto operand0 = data[0];
      auto operand1 = data[1];
      auto operand2 = data[2];
      assert(!operand0.inverted && !operand1.inverted && !operand2.inverted);
      assert(operand1.type == BitwiseOperandType::T &&
          operand2.type == BitwiseOperandType::T);
      if (operand0.type == BitwiseOperandType::T) {
        if (operand0.index == 0) {
          assert(operand1.index == 1 && operand2.index == 2);
          index = 12;
        } else {
          assert(operand0.index == 1 && operand1.index == 2 && operand2.index ==3);
          index = 13;
        }
      } else {
        if (operand0.index == 0) {
          assert(operand1.index == 1 && operand2.index == 2);
          index = 14;
        } else {
          assert(operand0.index == 1 && operand1.index == 0 && operand2.index == 3);
          index = 15;
        }
      }
      break;
    }

    default:
      return;
  }

  assert(index >= 0);
  addr.data = index;
  addr.str_repr = std::format("{}{}", 'B', index);
}
