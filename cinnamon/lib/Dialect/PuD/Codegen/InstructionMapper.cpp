#include "cinm-mlir/Dialect/PuD/Codegen/InstructionMapper.h"

#include "eggmock.hpp"
#include "gp.h"
#include "mockturtle/networks/mig.hpp"
#include "utils.h"

#include <sstream>
#include <stdexcept>
#include <string>

using namespace eggmock;
using MIG = mockturtle::mig_network;

namespace {

/// Renders a cell the way lime's own Display impl does ("T[0]", "DCC[1]",
/// "true"/"false" for constants), which is the syntax ProgramParser expects.
std::string cellStr(ffi_program const &prog, ffi_cell const &cell) {
  if (cell.cell_type_idx >= prog.num_cell_type_names)
    throw std::runtime_error("lime program references an unknown cell type");
  std::string type = prog.cell_type_names[cell.cell_type_idx];
  // the constant cell type is named "bool"; its index carries the value
  if (type == "bool")
    return cell.cell_index != 0 ? "true" : "false";
  return type + "[" + std::to_string(cell.cell_index) + "]";
}

} // namespace

/// Legacy bridge for the original BitsToPuD path, which consumes programs
/// through ProgramParser's text format.
///
/// The compiler FFI now hands back a structured instruction array rather than a
/// program string, so this re-renders that array into the text form the parser
/// still expects. New code should consume the structured export directly and
/// skip the round trip -- ProgramParser additionally hard-codes Ambit's operand
/// groups, which is why this keeps compiling for Ambit only.
std::string getProgram(MIG &mig) {
  preoptimize(mig);
  program_holder prog;
  compiler_settings settings{};
  settings.rewriting = rewriting_strategy::none;
  settings.validator = new_validator(mig);
  settings.mode = compilation_mode::exhaustive;
  settings.candidate_selection = candidate_selection_mode::all;
  settings.program_out = prog.out();

  auto result = send_ntk(mig, receiver(gp_compile_ambit(settings)));
  if (!result.validation_success)
    throw std::runtime_error("lime could not verify the compiled program");

  std::ostringstream out;
  for (auto const &instr : prog) {
    out << instr.opcode_name << "(";
    for (uint64_t i = 0; i < instr.num_inputs; i++)
      out << (i ? ", " : "") << cellStr(prog.get(), instr.inputs[i]);
    out << ")";
    if (instr.num_outputs > 0) {
      out << " -> (";
      for (uint64_t i = 0; i < instr.num_outputs; i++)
        out << (i ? ", " : "") << (instr.outputs[i].inverted ? "!" : "")
            << cellStr(prog.get(), instr.outputs[i].cell);
      out << ")";
    }
    out << "\n";
  }
  return out.str();
}
