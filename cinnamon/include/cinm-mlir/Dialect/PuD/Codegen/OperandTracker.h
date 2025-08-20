#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <optional>
#include <variant>


class OperandExpr {
public:
  virtual ~OperandExpr() = default;
  
  virtual std::unique_ptr<OperandExpr> clone() const = 0;
  virtual std::string toString() const = 0;
  virtual bool evaluate() const = 0;
};

class ConstantExpr : public OperandExpr {
private:
  bool bitVal;
  
public:
  ConstantExpr(bool bitVal) : bitVal(bitVal) {}
  
  std::unique_ptr<OperandExpr> clone() const override {
    return std::make_unique<ConstantExpr>(bitVal);
  }
  
  std::string toString() const override {
    return bitVal ? "C1" : "C0";
  }

  bool evaluate() const override {
    return bitVal;
  }
};

class DataExpr : public OperandExpr {
private:
  std::string name;
  bool bitVal;
  
public:
  DataExpr(const std::string &name, bool bitVal) : name(name), bitVal(bitVal) {}
  
  std::unique_ptr<OperandExpr> clone() const override {
    return std::make_unique<DataExpr>(name, bitVal);
  }
  
  std::string toString() const override {
    return name;
  }

  bool evaluate() const override {
    return bitVal;
  }
};

class NotExpr : public OperandExpr {
private:
  std::unique_ptr<OperandExpr> operand;
  
public:
  NotExpr(std::unique_ptr<OperandExpr> operand) : operand(std::move(operand)) {}
  
  std::unique_ptr<OperandExpr> clone() const override {
    return std::make_unique<NotExpr>(operand->clone());
  }
  
  std::string toString() const override {
    return "NOT(" + operand->toString() + ")";
  }

  bool evaluate() const override {
    return !operand->evaluate();
  }
};

class MajExpr : public OperandExpr {
private:
  std::unique_ptr<OperandExpr> operand1;
  std::unique_ptr<OperandExpr> operand2;
  std::unique_ptr<OperandExpr> operand3;
  
public:
  MajExpr(
      std::unique_ptr<OperandExpr> operand1,
      std::unique_ptr<OperandExpr> operand2,
      std::unique_ptr<OperandExpr> operand3)
      : operand1(std::move(operand1)),
        operand2(std::move(operand2)),
        operand3(std::move(operand3)) {}
  
  std::unique_ptr<OperandExpr> clone() const override {
    return std::make_unique<MajExpr>(
      operand1->clone(),
      operand2->clone(),
      operand3->clone()
    );
  }
  
  std::string toString() const override {
    return "MAJ(" + operand1->toString() + ", " + 
        operand2->toString() + ", " + 
        operand3->toString() + ")";
  }

  bool evaluate() const override {
    const auto a = operand1->evaluate();
    const auto b = operand2->evaluate();
    const auto c = operand3->evaluate();
    return (a && b) || (a && c) || (b && c);
  }
};

class OperandTracker {
private:
  llvm::SmallVector<std::unique_ptr<OperandExpr>, 16> bGroupExpressions;

  std::unique_ptr<OperandExpr> doMaj(int index1, int index2, int index3);
  void doNot(int index);

public:
  explicit OperandTracker();
  std::unique_ptr<OperandExpr> executeAP(int index);
  std::unique_ptr<OperandExpr> executeAAP(
      std::variant<OperandExpr *, int> source,
      std::optional<int> destination);
  llvm::ArrayRef<std::unique_ptr<OperandExpr>> getBGroupExprs() const {
    return bGroupExpressions;
  }
};
