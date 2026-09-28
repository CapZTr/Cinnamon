// Row-op primitives for --pud-primitives: the networks lime compiles into
// AAP/AP sequences, read by cinm-opt (and rowop-sanity) instead of the
// built-in definitions in RowOpLibrary.cpp. Editing this file changes what a
// primitive costs and what it emits without rebuilding cinm-opt:
//
//   cinm-opt --pud-primitives=full_adder.mlir kernel.mlir --convert-linalg-to-bits ...
//   rowop-sanity --pud-primitives=full_adder.mlir     # the sequence, its cost
//
// This is the full adder exactly as built in: carry-in and carry-out stay in
// dual-contact cell 1 from one bit to the next, and lime finds 8 commands.
// Remove the two pud.resident attributes and it takes 9 -- and the kernels
// that rely on a resident carry are refused, which is the point of the hint.
//
// A primitive's network has to compute that primitive: every definition is
// checked on a model of the substrate and refused if it does not. The format
// is described in cinnamon/lib/Dialect/Bits/Codegen/PrimitiveFile.cpp.
func.func @row_add(%a: !bits.row<1>, %b: !bits.row<1>, %cin: !bits.row<1> {pud.resident = "DCC1"})
    -> (!bits.row<1>, !bits.row<1> {pud.resident = "DCC1"})
    attributes {pud.commands = 8} {
  %cout  = bits.maj %a, %b, %cin : !bits.row<1>, !bits.row<1>, !bits.row<1> -> !bits.row<1>
  %nb    = bits.not %b : !bits.row<1> -> !bits.row<1>
  %m     = bits.maj %a, %nb, %cin : !bits.row<1>, !bits.row<1>, !bits.row<1> -> !bits.row<1>
  %ncout = bits.not %cout : !bits.row<1> -> !bits.row<1>
  %sum   = bits.maj %b, %ncout, %m : !bits.row<1>, !bits.row<1>, !bits.row<1> -> !bits.row<1>
  return %sum, %cout : !bits.row<1>, !bits.row<1>
}
