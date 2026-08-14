//===- Attributes.td - PuD dialect attributes --------------*- tablegen -*-===//
//
// This is the definitions file for the PuD dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef PUD_ATTRIBUTES
#define PUD_ATTRIBUTES

include "PuDBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Substrate
//===----------------------------------------------------------------------===//

// Which processing-using-DRAM substrate an op belongs to.
//
// The substrates do not share a command set: SIMDRAM computes with
// triple-row activation and dual-contact cells, while PRADA activates rows
// sequentially, gets a three-row majority in one command, and has no dual-contact
// cell to read a complement from. An op is therefore only meaningful for the
// substrate it was written for, and says which one that is.
def PuD_Backend_Simdram : I32EnumAttrCase<"Simdram", 0, "simdram">;

def PuD_Backend : I32EnumAttr<
    "Backend", "processing-using-DRAM substrate",
    [PuD_Backend_Simdram]> {
  let cppNamespace = "::mlir::pud";
  let genSpecializedAttr = 0;
}

def PuD_BackendAttr : EnumAttr<PuD_Dialect, PuD_Backend, "backend">;

//===----------------------------------------------------------------------===//
// SIMDRAM control rows
//===----------------------------------------------------------------------===//

// The reserved rows a SIMDRAM subarray computes with.
//
// Four compute rows (T0-T3) and two dual-contact cells, each of which presents
// its value on one row and its complement on another (DCC0 / DCC0N). The
// remaining names are the row sets a single activation drives: the dual-row
// pairs, and the triples whose activation leaves the majority of the three in
// every one of them.
//
// Which sets exist is a property of the substrate -- they are wired that way --
// so they are named here rather than computed. Two constant rows, held at 0 and
// at 1, complete the set.
def PuD_Ctrl_T0          : I32EnumAttrCase<"T0",          0, "T0">;
def PuD_Ctrl_T1          : I32EnumAttrCase<"T1",          1, "T1">;
def PuD_Ctrl_T2          : I32EnumAttrCase<"T2",          2, "T2">;
def PuD_Ctrl_T3          : I32EnumAttrCase<"T3",          3, "T3">;
def PuD_Ctrl_DCC0        : I32EnumAttrCase<"DCC0",        4, "DCC0">;
def PuD_Ctrl_DCC0N       : I32EnumAttrCase<"DCC0N",       5, "DCC0N">;
def PuD_Ctrl_DCC1        : I32EnumAttrCase<"DCC1",        6, "DCC1">;
def PuD_Ctrl_DCC1N       : I32EnumAttrCase<"DCC1N",       7, "DCC1N">;
def PuD_Ctrl_DCC0N_T0    : I32EnumAttrCase<"DCC0N_T0",    8, "DCC0N_T0">;
def PuD_Ctrl_DCC1N_T1    : I32EnumAttrCase<"DCC1N_T1",    9, "DCC1N_T1">;
def PuD_Ctrl_T2_T3       : I32EnumAttrCase<"T2_T3",      10, "T2_T3">;
def PuD_Ctrl_T0_T3       : I32EnumAttrCase<"T0_T3",      11, "T0_T3">;
def PuD_Ctrl_T0_T1_T2    : I32EnumAttrCase<"T0_T1_T2",   12, "T0_T1_T2">;
def PuD_Ctrl_T0_T1_T3    : I32EnumAttrCase<"T0_T1_T3",   13, "T0_T1_T3">;
def PuD_Ctrl_DCC0_T1_T3  : I32EnumAttrCase<"DCC0_T1_T3", 14, "DCC0_T1_T3">;
def PuD_Ctrl_DCC1_T0_T2  : I32EnumAttrCase<"DCC1_T0_T2", 15, "DCC1_T0_T2">;
def PuD_Ctrl_C_0         : I32EnumAttrCase<"C_0",        16, "C_0">;
def PuD_Ctrl_C_1         : I32EnumAttrCase<"C_1",        17, "C_1">;

def PuD_SimdramCtrlRow : I32EnumAttr<
    "SimdramCtrlRow", "reserved control row of a SIMDRAM subarray",
    [PuD_Ctrl_T0, PuD_Ctrl_T1, PuD_Ctrl_T2, PuD_Ctrl_T3,
     PuD_Ctrl_DCC0, PuD_Ctrl_DCC0N, PuD_Ctrl_DCC1, PuD_Ctrl_DCC1N,
     PuD_Ctrl_DCC0N_T0, PuD_Ctrl_DCC1N_T1, PuD_Ctrl_T2_T3, PuD_Ctrl_T0_T3,
     PuD_Ctrl_T0_T1_T2, PuD_Ctrl_T0_T1_T3, PuD_Ctrl_DCC0_T1_T3,
     PuD_Ctrl_DCC1_T0_T2, PuD_Ctrl_C_0, PuD_Ctrl_C_1]> {
  let cppNamespace = "::mlir::pud";
  let genSpecializedAttr = 0;
}

def PuD_SimdramCtrlRowAttr
    : EnumAttr<PuD_Dialect, PuD_SimdramCtrlRow, "ctrl_row">;

#endif
