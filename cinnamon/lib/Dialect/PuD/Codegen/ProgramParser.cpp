#include "cinm-mlir/Dialect/PuD/Codegen/ProgramParser.h"

#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <llvm/Support/LogicalResult.h>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>


ProgramParser::ProgramParser(std::string input) : input(input) {}

llvm::LogicalResult ProgramParser::parse() {
  while (position < input.size()) {
    size_t lineStart = position;
    size_t lineEnd = input.find('\n', position);
    if (lineEnd == std::string::npos) {
      lineEnd = input.size();
    }
    size_t i = lineStart;
    while (i < lineEnd && std::isspace(input[i] && input[i] != '\n')) {
      ++i;
    }
    if ((i >= lineEnd) || (i + 1 < lineEnd && input[i] == '/' && input[i + 1] == '/')) {
      position = (lineEnd < input.size()) ? lineEnd + 1 : lineEnd;
      currentLine++;
      continue;
    }
    std::string instr;
    while (std::isalpha(peek())) {
      instr += peek();
      advance();
    }
    assert(!instr.empty());
    Instruction instruction;
    if (instr == "RC") {
      instruction = parseRC(lineEnd);
    } else {
      assert(instr == "TRA");
      instruction = parseTRA(lineEnd);
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
  while (position < input.size() && std::isspace(input[position]) && input[position] != '\n') {
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

bool ProgramParser::match(char c) {
  if (peek() == c) {
    advance();
    return true;
  }
  return false;
}

int ProgramParser::parseNumber() {
  int val = 0;
  while (std::isdigit(peek())) {
    val = val * 10 + (peek() - '0');
    advance();
  }
  return val;
}

Instruction ProgramParser::parseRC(size_t lineEnd) {
  Instruction inst;
  bool inverted = false;
  if (match('_')) {
    assert(input.substr(position, 3) == "INV");
    position += 3;
    inverted = true;
  }
  assert(match('(') && "expected '(' after RC");
  if (inverted) {
    assert(input.substr(position, 3) == "DCC");
  }
  inst.type = Instruction::Type::AAP;
  size_t srcEnd = input.find(')', position);
  assert(srcEnd != std::string::npos);
  inst.operand0 = parseAddress(srcEnd);
  if (inverted) {
    auto index = std::get<int>(inst.operand0.data);
    if (index == 4) {
      inst.operand0.data = 5;
      inst.operand0.str_repr = "B5";
    } else {
      assert(index == 6);
      inst.operand0.data = 7;
      inst.operand0.str_repr = "B7";
    }
  }
  assert(position == srcEnd && match(')'));
  skipWhitespace();
  assert(match('-'));
  assert(match('>'));
  skipWhitespace();
  assert(match('('));
  size_t dstEnd = input.find(')', position);
  assert(dstEnd != std::string::npos);
  inst.operand1 = parseAddress(dstEnd);
  assert(match(')'));
  assert(position == lineEnd);
  return inst;
}

Instruction ProgramParser::parseTRA(size_t lineEnd) {
  assert(match('(') && "expected '(' after TRA");
  Instruction inst;
  size_t traEnd = input.find(')', position);
  assert(traEnd != std::string::npos);
  inst.operand0 = parseAddress(traEnd);
  assert(position == traEnd && match(')'));
  if (position == lineEnd) {
    inst.type = Instruction::Type::AP;
    inst.operand1 = std::nullopt;
  } else {
    skipWhitespace();
    assert(match('-'));
    assert(match('>'));
    skipWhitespace();
    assert(match('('));
    size_t dstEnd = input.find(')', position);
    assert(dstEnd != std::string::npos);
    inst.operand1 = parseAddress(dstEnd);
    assert(match(')'));
    inst.type = Instruction::Type::AAP;
  }
  return inst;
}

Address ProgramParser::parseAddress(size_t addrEnd) {
  Address addr;
  if (match('D')) {
    if (match('[')) {
      int index = parseNumber();
      assert(match(']') && position == addrEnd);
      addr.type = AddressType::Data;
      addr.data = index;
      addr.str_repr = "D" + std::to_string(index);
      return addr;
    } else {
      position--;
      addr.type = AddressType::Bitwise;
      addr.data = parseBitwiseOperands(addrEnd);
      normalizeBitwiseAddress(addr);
      assert(position == addrEnd);
      return addr;
    }
  } else if (std::strncmp(&input[position], "true", 4) == 0) {
    position+=4;
    addr.type = AddressType::Const;
    addr.data = true;
    addr.str_repr = "C1";
    assert(position == addrEnd);
    return addr;
  } else if (std::strncmp(&input[position], "false", 5) == 0) {
    position+=5;
    addr.type = AddressType::Const;
    addr.data = false;
    addr.str_repr = "C0";
    assert(position == addrEnd);
    return addr;
  } else {
    assert(match('T') || match('!'));
    position--;
    addr.type = AddressType::Bitwise;
    addr.data = parseBitwiseOperands(addrEnd);
    normalizeBitwiseAddress(addr);
    assert(position == addrEnd);
    return addr;
  }
}

std::vector<BitwiseOperand> ProgramParser::parseBitwiseOperands(size_t addrEnd) {
  std::vector<BitwiseOperand> operands;
  while (position < addrEnd) {
    skipWhitespace();
    assert(match('T') || match('D') || match('!'));
    position--;
    operands.emplace_back(parseBitwiseOperand());
    if (position < addrEnd) {
      assert(match(','));
      assert(match(' '));
      assert(position < addrEnd);
    }
  }
  return operands;
}

BitwiseOperand ProgramParser::parseBitwiseOperand() {
  BitwiseOperand operand;
  if (match('T')) {
    operand.type = BitwiseOperandType::T;
    assert(match('['));
    operand.index = parseNumber();
    operand.inverted = false;
    assert(match(']'));
  } else if (input.substr(position, 3) == "DCC") {
    position += 3;
    assert(match('['));
    operand.type = BitwiseOperandType::DCC;
    operand.index = parseNumber();
    operand.inverted = false;
    assert(match(']'));
  } else {
    assert(input.substr(position, 4) == "!DCC");
    position+=4;
    assert(match('['));
    operand.type = BitwiseOperandType::DCC;
    operand.index = parseNumber();
    operand.inverted = true;
    assert(match(']'));
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
  addr.str_repr = "B" + std::to_string(index);
}
