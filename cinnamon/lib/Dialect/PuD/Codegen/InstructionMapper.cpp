#include "cinm-mlir/Dialect/PuD/Codegen/InstructionMapper.h"

#include "eggmock.hpp"
#include "gp.h"
#include "mockturtle/networks/mig.hpp"
#include "utils.h"

using namespace eggmock;
using MIG = mockturtle::mig_network;


std::string getProgram(MIG &mig) {
  preoptimize(mig);
  ProgramStringGP program_str;
  auto result = send_ntk(mig, receiver(gp_compile_ambit_with_program(compiler_settings{
      .rewriting = rewriting_strategy::none,
      .validator = new_validator(mig),
      .mode = compilation_mode::exhaustive,
      .candidate_selection = candidate_selection_mode::all,})));
  if (result.program_str) {
    program_str = ProgramStringGP(const_cast<char*>(result.program_str));
    result.program_str = nullptr;
  }
  return program_str.str();
}
