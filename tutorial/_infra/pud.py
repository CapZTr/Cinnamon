"""Helpers for the PuD tutorial notebooks.

Every stage of the compiled PuD path, one function each, so that a notebook can
run a stage, show what it wrote, and move on. Nothing here hides a step: each
function runs one command, saves its output under ``tutorial/artifacts/`` and
returns the path.

Tools are found in this order: an environment variable, the place the tutorial
image puts them, then ``PATH``.

    CINM_OPT      cinm-opt
    LLVM_BIN      directory holding mlir-translate, opt and llc
    PUD_CC        a gcc whose static binaries gem5's syscall emulation can run
                  (gcc 4.8 against Ubuntu 14.04's C library; see the Dockerfile)
    GEM5_PUD      gem5 built for X86_PUD
    MIMDRAM_ROOT  the MIMDRAM tree: the guest runtime and gem5's configs
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    env = os.environ.get("CINM_ROOT")
    if env:
        return Path(env)
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / "tutorial" / "_infra").is_dir():
            return p
    raise RuntimeError("cannot find the repository root; set CINM_ROOT")


ROOT = _repo_root()
ARTIFACTS = ROOT / "tutorial" / "artifacts"
KERNELS = ROOT / "limon_benchmark"
MIMDRAM = Path(os.environ.get("MIMDRAM_ROOT", ROOT / "third-party" / "MIMDRAM"))


def _tool(env: str, *candidates: Path | str) -> str:
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        c = Path(c)
        if c.is_absolute() and c.exists():
            return str(c)
        found = shutil.which(str(c))
        if found:
            return found
    raise FileNotFoundError(f"cannot find {candidates[-1]}; set {env}")


def cinm_opt() -> str:
    return _tool("CINM_OPT", ROOT / "cinnamon" / "build" / "bin" / "cinm-opt",
                 "cinm-opt")


def llvm(tool: str) -> str:
    base = os.environ.get("LLVM_BIN")
    if base:
        return str(Path(base) / tool)
    return _tool("LLVM_BIN", Path("/opt/llvm/bin") / tool, tool)


def rowop_sanity() -> str:
    return _tool("ROWOP_SANITY", Path(cinm_opt()).parent / "rowop-sanity",
                 "rowop-sanity")


def pud_cc() -> str:
    return _tool("PUD_CC", "pud-gcc")


def gem5() -> str:
    return _tool("GEM5_PUD", "gem5-pud")


# --------------------------------------------------------------------------
# Running things
# --------------------------------------------------------------------------

@dataclass
class Result:
    """What a command printed, and how long it took."""
    stdout: str
    stderr: str
    seconds: float


def run(cmd: list[str], *, cwd: Path | None = None) -> Result:
    """Runs `cmd`, raising with its output if it fails."""
    start = time.time()
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    took = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}")
    return Result(proc.stdout, proc.stderr, took)


def artifact(name: str) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / name


def show(path: Path, lines: int = 40, grep: str | None = None) -> None:
    """Prints the start of a file, or only the lines matching `grep`."""
    text = Path(path).read_text().splitlines()
    if grep:
        text = [l for l in text if re.search(grep, l)]
    print(f"--- {Path(path).relative_to(ROOT)} "
          f"({len(Path(path).read_text().splitlines())} lines) ---")
    for l in text[:lines]:
        print(l)
    if len(text) > lines:
        print(f"... {len(text) - lines} more")


# --------------------------------------------------------------------------
# The Gurobi licence
# --------------------------------------------------------------------------

def licence_file() -> Path | None:
    """The licence Gurobi will use, if there is one where it looks."""
    for c in (os.environ.get("GRB_LICENSE_FILE"), Path.home() / "gurobi.lic",
              "/opt/gurobi/gurobi.lic"):
        if c and Path(c).is_file():
            return Path(c)
    return None


class LicenceError(RuntimeError):
    pass


def _check_gurobi(stderr: str, stdout: str) -> None:
    # cinm-opt exits 0 when Gurobi refuses to run and only says so on its
    # output, so this is the one place that notices.
    for text in (stderr, stdout):
        m = re.search(r"Gurobi error: (.*)", text)
        if m:
            raise LicenceError(
                f"the mapping pass could not use Gurobi: {m.group(1)}\n"
                "It needs a licence the C++ API accepts -- an academic WLS "
                "licence works in this image. Upload gurobi.lic to your home "
                "directory (see 00_intro).")


# --------------------------------------------------------------------------
# The stages
# --------------------------------------------------------------------------

MEMORIES = {
    # name: (gem5 --mem-type, banks per channel)
    "ddr": ("DDR4_2400_x64", 32),
    "hbm": ("HBM3_4Gb_x64", 16),
}


def kernel(name: str) -> Path:
    """A kernel from limon_benchmark/, by the part of its name after linalg_."""
    path = KERNELS / f"linalg_{name}.mlir"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def to_bits(src: Path, tag: str) -> Path:
    """Step 1: linalg on tensors to bit-sliced operations."""
    out = artifact(f"{tag}_1_bits.mlir")
    run([cinm_opt(), src, "--convert-linalg-to-bits", "-o", out])
    return out


@dataclass
class Mapping:
    path: Path
    objective: float | None
    model: str | None
    seconds: float
    trace: Path | None


def map_banks(bits: Path, tag: str, banks: int, mem: str = "ddr",
              trace: bool = False, flags: tuple[str, ...] = ()) -> Mapping:
    """Step 2: the ILP chooses how loops spread over banks.

    With `trace`, the pass also writes the CIMTRACE file the trace player
    replays -- the paper's path -- next to the other artifacts.
    """
    out = artifact(f"{tag}_2_mapped.mlir")
    trace_path = artifact(f"{tag}.trace") if trace else None
    opts = f"num-banks={banks} mem-type={mem} emit-trace={'true' if trace else 'false'}"
    if trace_path:
        opts += f" trace-path={trace_path}"
    r = run([cinm_opt(), *flags, bits, f"--bits-opt-mapping={opts}", "-o", out])
    _check_gurobi(r.stderr, r.stdout)
    log = r.stdout + r.stderr
    obj = re.search(r"Best objective ([-+0-9.e]+)", log)
    model = re.search(r"Optimize a model with (.*)", log)
    return Mapping(out, float(obj.group(1)) if obj else None,
                   model.group(1).strip() if model else None, r.seconds,
                   trace_path)


def mapping_attrs(mapped: Path) -> list[dict]:
    """The mapping's decisions: the bits.* attributes on each mapped loop.

    A reduction also carries `reduce_tree`: one list of (source bank,
    destination bank) copies per level of the tree that folds the banks'
    partial results together.
    """
    text = Path(mapped).read_text()
    loops = []
    for start in [m.start() for m in re.finditer(r"\{bits\.bank_group_stride", text)]:
        depth, end = 0, start
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                end = i
                break
        block = text[start:end + 1]
        attrs = {}
        tree = re.search(r"bits\.reduce_tree = (\[.*?\]\])", block, re.S)
        if tree:
            attrs["reduce_tree"] = [
                [(int(s), int(d)) for d, s in re.findall(
                    r"dst_bank = (\d+) : i64, src_bank = (\d+)", level)]
                for level in re.findall(r"\[(\{.*?\})\]", tree.group(1))]
            block = block.replace(tree.group(0), "")
        for k, v in re.findall(r"bits\.(\w+) = (\"[^\"]*\"|-?\d+)", block):
            attrs[k] = v.strip('"') if v.startswith('"') else int(v)
        loops.append(attrs)
    return loops


def expand_rows(mapped: Path, tag: str, flags: tuple[str, ...] = ()) -> Path:
    """Step 3: slice operations become loops over single rows."""
    out = artifact(f"{tag}_3_rows.mlir")
    run([cinm_opt(), *flags, mapped, "--bits-pattern-apply", "-o", out])
    return out


def to_pud(rows: Path, tag: str, resident: bool = True,
           flags: tuple[str, ...] = ()) -> Path:
    """Step 4: row operations become DRAM commands (AAP/AP), bank by bank."""
    out = artifact(f"{tag}_4_pud.mlir")
    run([cinm_opt(), *flags, rows,
         f"--convert-bits-rows-to-pud=operand-per-wave={1 if resident else 0}",
         "-o", out])
    return out


def row_space(pud: Path) -> dict:
    """The shape of the row space the compiler published (pud.* attributes)."""
    text = Path(pud).read_text()
    return {k: int(v) for k, v in re.findall(r"pud\.(\w+) = (\d+)", text)}


def to_llvm_dialect(pud: Path, tag: str) -> Path:
    """Step 5: addresses and inline instructions, then the stock way out of MLIR."""
    out = artifact(f"{tag}_5_llvm.mlir")
    run([cinm_opt(), pud, "--convert-pud-rows-to-func", "--convert-scf-to-cf",
         "--convert-arith-to-llvm", "--convert-func-to-llvm",
         "--reconcile-unrealized-casts", "-o", out])
    return out


def to_object(llvm_mlir: Path, tag: str, banks: int, mem: str = "ddr") -> Path:
    """Step 6: LLVM IR, the optimiser and the backend, as the build script does."""
    ll = artifact(f"{tag}_6_kernel.ll")
    bc = artifact(f"{tag}_6_kernel.opt.bc")
    obj = artifact(f"{tag}_6_kernel.o")
    run([llvm("mlir-translate"), "--mlir-to-llvmir", llvm_mlir, "-o", ll])
    run([llvm("opt"), "-O3", "-unroll-threshold=0", ll, "-o", bc])
    align = []
    if banks > MEMORIES[mem][1]:
        # Past one channel the row-op loops are folded into one, and its
        # alignment decides the fetch cost (HANDOVER.md, section 4).
        align = ["--x86-experimental-pref-innermost-loop-alignment=6"]
    run([llvm("llc"), "-filetype=obj", "-relocation-model=static", *align, bc,
         "-o", obj])
    return obj


ROW_OPS = {0x60: "AP", 0x61: "AAP", 0x62: "ROWCOPY", 0x6f: "ROW_RD_STREAM",
           0x70: "ROW_WR_STREAM"}


def first_row_op_loop(obj: Path) -> str:
    """The first loop in `obj` that issues a row op, readably.

    objdump does not know the row-op instruction -- 0F 04 and a two-byte
    selector -- and loses sync for several instructions after it. So the row op
    is printed from its bytes, and the rest of the loop is disassembled again
    from the byte after it, up to the branch that closes the loop.
    """
    dis = run(["objdump", "-d", obj]).stdout.splitlines()
    at = next(l for l in dis if re.match(r"\s*[0-9a-f]+:\s+0f 04\s", l))
    addr = int(at.split(":")[0], 16)
    sel = next(l for l in dis if re.match(rf"\s*{addr + 2:x}:", l))
    op = int(sel.split("\t")[1].split()[0], 16)
    def decode(start: int, stop: int, until_branch: bool) -> list[str]:
        out = []
        for l in run(["objdump", "-d", f"--start-address={start}",
                      f"--stop-address={stop}", obj]).stdout.splitlines():
            if not re.match(r"\s*[0-9a-f]+:\s", l):
                continue
            parts = l.split("\t")
            out.append(f"{parts[0].strip():>7}  {parts[1].strip():20} "
                       f"{' '.join(parts[2:]).strip()}")
            if until_branch and re.search(r"\bj[a-z]+\s+[0-9a-f]+ <", l):
                break
        return out

    after = decode(addr + 4, addr + 64, until_branch=True)
    # The loop starts where its closing branch goes back to, which can be a
    # few instructions before the row op: those compute its addresses.
    target = re.search(r"\bj[a-z]+\s+([0-9a-f]+) <", after[-1])
    start = int(target.group(1), 16) if target else addr
    before = decode(start, addr, until_branch=False) if start < addr else []
    row_op = f"{addr:x}:".rjust(7) + f"  {'0f 04 %02x 00' % op:20} rowop {ROW_OPS.get(op, hex(op))}"
    return "\n".join(before + [row_op] + after)


def kernel_symbol(pud: Path) -> str:
    m = re.search(r"func\.func @([A-Za-z_]\w*)", Path(pud).read_text())
    if not m:
        raise RuntimeError(f"no kernel function in {pud}")
    return m.group(1)


def link(obj: Path, pud: Path, tag: str) -> Path:
    """Step 7: the kernel with the guest runtime, into a static binary."""
    space = row_space(pud)
    micro = MIMDRAM / "microworkloads"
    exe = artifact(f"{tag}.exe")
    run([pud_cc(), "-static", "-std=c99", "-O2", "-msse2",
         "-I", MIMDRAM / "gem5" / "util" / "m5", "-I", micro,
         f"-DPUD_KERNEL={kernel_symbol(pud)}",
         f"-DPUD_DATA_ROWS={space['data_rows_per_bank']}",
         f"-DPUD_BANKS_PER_CHANNEL={space['banks_per_channel']}",
         f"-DPUD_TOTAL_BANKS={space['total_banks']}",
         f"-DPUD_CHANNEL_SIZE={space['channel_size']}",
         f"-DPUD_ROW_SIZE={space['row_stride']}",
         micro / "m5op_x86.S", micro / "rowop.S",
         micro / "pud" / "pud_main.c", micro / "pud" / "pud_runtime.c",
         obj, "-o", exe])
    return exe


@dataclass
class Kernel:
    """Everything one compilation wrote."""
    tag: str
    banks: int
    mem: str
    bits: Path
    mapping: Mapping
    rows: Path
    pud: Path
    llvm_mlir: Path
    obj: Path
    exe: Path
    resident: bool


def compile_kernel(name: str, banks: int, mem: str = "ddr",
                   trace: bool = False, flags: tuple[str, ...] = (),
                   tag: str | None = None) -> Kernel:
    """All of steps 1 to 7 for a limon_benchmark kernel.

    A kernel whose operand does not fit in a subarray at this bank count is
    built with its waves sharing rows -- a faithful model of the schedule's
    cost, but not a program -- exactly as compile_pud_binary.sh falls back.

    `flags` are cinm-opt options every stage has to see, such as
    --pud-primitives: the mapping prices a primitive, the expansion and the
    emitter use it, and they have to agree. `tag` names the artifacts.
    """
    tag = tag or f"{name}_{mem}_{banks}"
    flags = tuple(flags)
    bits = to_bits(kernel(name), tag)
    mapping = map_banks(bits, tag, banks, mem, trace=trace, flags=flags)
    rows = expand_rows(mapping.path, tag, flags=flags)
    resident = True
    try:
        pud = to_pud(rows, tag, resident=True, flags=flags)
    except RuntimeError:
        resident = False
        pud = to_pud(rows, tag, resident=False, flags=flags)
    llvm_mlir = to_llvm_dialect(pud, tag)
    obj = to_object(llvm_mlir, tag, banks, mem)
    exe = link(obj, pud, tag)
    return Kernel(tag, banks, mem, bits, mapping, rows, pud, llvm_mlir, obj,
                  exe, resident)


# --------------------------------------------------------------------------
# Simulating
# --------------------------------------------------------------------------

@dataclass
class Stats:
    row_ops: int
    makespan_us: float
    seconds: float
    outdir: Path
    extra: dict = field(default_factory=dict)


def _read_stats(stats: Path, first_block: bool = True) -> dict:
    blocks = Path(stats).read_text().split("Begin Simulation Statistics")
    block = blocks[1] if first_block and len(blocks) > 1 else blocks[-1]
    values = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return values


def simulate(exe: Path, mem: str = "ddr", cpu: str = "o3",
             cpu_clock: str = "2GHz") -> Stats:
    """Runs a compiled kernel on gem5 and reads the measured run.

    The driver runs the kernel twice and gem5 dumps statistics twice, so only
    the first block is the measured run; row-op counts are per controller and
    are summed; a controller that saw no row op reports a first tick of zero,
    which is skipped.
    """
    out = artifact(f"m5out_{Path(exe).stem}_{cpu_clock}")
    r = run([gem5(), f"--outdir={out}",
             MIMDRAM / "gem5" / "configs" / "dram" / "cpu_rowop.py",
             f"--cmd={exe}", f"--mem-type={MEMORIES[mem][0]}",
             f"--cpu-type={cpu}", f"--cpu-clock={cpu_clock}"])
    v = _read_stats(out / "stats.txt")
    ops = int(sum(x for k, x in v.items() if k.endswith("rowOpCount")))
    first = [x for k, x in v.items() if k.endswith("rowOpFirstTick") and x > 0]
    last = [x for k, x in v.items() if k.endswith("rowOpLastTick")]
    span = (max(last) - min(first)) / 1e6 if first else 0.0
    return Stats(ops, span, r.seconds, out,
                 {"committed_insts": v.get("system.cpu.commit.committedInsts"),
                  "cycles": v.get("system.cpu.numCycles")})


def replay(trace: Path, mem: str = "ddr") -> Stats:
    """Replays a CIMTRACE file with gem5's trace player -- the paper's path."""
    out = artifact(f"m5out_{Path(trace).stem}_trace")
    r = run([gem5(), f"--outdir={out}",
             MIMDRAM / "gem5" / "configs" / "dram" / "trace_player.py",
             f"--trace={trace}", f"--mem-type={MEMORIES[mem][0]}",
             "--backend=simdram"])
    v = _read_stats(out / "stats.txt", first_block=False)
    ops = int(sum(x for k, x in v.items() if k.endswith("rowOpCount")))
    return Stats(ops, v.get("system.player.rowOpMakespan", 0.0) / 1e6,
                 r.seconds, out)

# --------------------------------------------------------------------------
# Primitives, and a model of what an approximate addition computes
# --------------------------------------------------------------------------

def primitive_table(flags: tuple[str, ...] = (), names: tuple[str, ...] = ()) -> str:
    """rowop-sanity's table and sequences, for these options.

    With `names`, only those primitives' rows of the table and their sequences.
    """
    extra = [f for f in flags if f.startswith("--pud-primitives=")]
    out = run([rowop_sanity(), *extra]).stdout
    if not names:
        return out
    lines = out.splitlines()
    keep = [l for l in lines if l.startswith("primitive")][:1]
    keep += [l for l in lines if l.split() and l.split()[0] in names]
    for name in names:
        if f"{name}:" in lines:
            i = lines.index(f"{name}:")
            j = next((k for k in range(i + 1, len(lines)) if not lines[k].strip()),
                     len(lines))
            keep += [""] + lines[i:j]
    return "\n".join(keep)


def network_function(path: Path, primitive: str):
    """The function a primitive's network in `path` computes, on integers.

    It evaluates the same MLIR the compiler reads -- bits.maj, bits.not and the
    constant rows -- bitwise, so it works on whole words (or numpy arrays) at a
    time: bit i of the result is the network applied to bit i of the operands.
    """
    text = Path(path).read_text()
    m = re.search(rf"func\.func @{primitive}\((.*?)\)\s*->.*?\{{(.*?)\n\}}",
                  text, re.S)
    if not m:
        raise ValueError(f"no @{primitive} in {path}")
    args = re.findall(r"(%\w+)\s*:", m.group(1))
    body = m.group(2)

    def f(*operands, mask=(1 << 64) - 1):
        env = dict(zip(args, operands))
        for line in body.splitlines():
            line = line.strip()
            if (d := re.match(r"(%\w+)\s*=\s*bits\.maj\s+(%\w+),\s*(%\w+),\s*(%\w+)", line)):
                x, y, z = (env[v] for v in d.group(2, 3, 4))
                env[d.group(1)] = (x & y) | (x & z) | (y & z)
            elif (d := re.match(r"(%\w+)\s*=\s*bits\.not\s+(%\w+)", line)):
                env[d.group(1)] = ~env[d.group(2)] & mask
            elif (d := re.match(r"(%\w+)\s*=\s*bits\.create_all_zero_row", line)):
                env[d.group(1)] = 0 * operands[0]
            elif (d := re.match(r"(%\w+)\s*=\s*bits\.create_all_one_row", line)):
                env[d.group(1)] = 0 * operands[0] + mask
            elif (d := re.match(r"return\s+(.*?)\s*:", line)):
                outs = [env[v.strip()] for v in d.group(1).split(",")]
                return outs[0] if len(outs) == 1 else tuple(outs)
        raise ValueError(f"@{primitive} in {path} does not return")
    return f


def approx_add(a, b, width: int, low_bits: int, cell):
    """What --pud-approx-add-low-bits computes, as the compiler expands it.

    Bits [0, low_bits) are `cell` applied bitwise to the operands' low bits;
    bits [low_bits, width) are the exact sum of the high parts with carry 0.
    Works on Python ints and on numpy integer arrays.
    """
    low_mask = (1 << low_bits) - 1
    high = ((a >> low_bits) + (b >> low_bits)) << low_bits
    low = cell(a & low_mask, b & low_mask, mask=low_mask) & low_mask
    return (high | low) & ((1 << width) - 1)
