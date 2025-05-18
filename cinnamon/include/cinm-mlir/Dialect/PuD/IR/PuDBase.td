//===- Base.td - PuD dialect base --------------------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_BASE
#define PUD_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def PuD_Dialect : Dialect {
  let name = "pud";
  let cppNamespace = "::mlir::pud";

  let summary = "TODO";
  let description = [{
    TODO
  }];

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
  private:
    void registerOps();
    void registerTypes();
  }];
}

// Template for attributes.
class PuD_Attr<string name, list<Trait> traits = []>
    : AttrDef<PuD_Dialect, name, traits>;

// Template for ops.
class PuD_Op<string mnemonic, list<Trait> traits = []>
    : Op<PuD_Dialect, mnemonic, traits>;

// Template for types.
class PuD_Type<string name, list<Trait> traits = []>
    : TypeDef<PuD_Dialect, name, traits>;

#endif
