//===- Base.td - Bits dialect base -------------------------*- tablegen -*-===//
//
// This is the definitions file for the Bits dialect base.
//
//===----------------------------------------------------------------------===//

#ifndef BITS_BASE
#define BITS_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def Bits_Dialect : Dialect {
    let name = "bits";
    let cppNamespace = "::mlir::bits";

    let summary = "TODO";
    let description = [{
        TODO
    }];

    let useDefaultTypePrinterParser = 1;

    code extraClassDeclaration = [{
    private:
        void registerOps();
        void registerTypes();
    }];
}

// Template for attributes.
class Bits_Attr<string name, list<Trait> traits = []>
        : AttrDef<Bits_Dialect, name, traits>;

// Template for ops.
class Bits_Op<string mnemonic, list<Trait> traits = []>
        : Op<Bits_Dialect, mnemonic, traits>;

// Template for types.
class Bits_Type<string name, list<Trait> traits = []>
        : TypeDef<Bits_Dialect, name, traits>;

#endif
