# The PuD compilation flow

What the two repositories do, how a kernel gets from `linalg` to a running
simulation, and what state that path is in. Written 2026-09-22.

Everything here concerns two repositories and nothing else:

| repo | branch | commit | role |
|---|---|---|---|
| **Cinnamon** | `pud-codegen` | `104f182` | the compiler, and the kernels it is measured on |
| **MIMDRAM** | `pud-codegen` | `4385f03` | the guest runtime, the ISA extension, and gem5 |

Both branches are pushed and are what you should check out. Their bases,
Cinnamon `simdram-cpp-20` and MIMDRAM `main`, are the paper's versions and are
untouched by any of this.

---

## 1. What changed, in one paragraph

The paper's version of this pipeline **stops at a trace**. `cinm-opt` lowers a
kernel to bit-serial operations, an ILP maps its loops onto banks, and the
mapping pass writes out the sequence of DRAM row operations that schedule would
issue. gem5 then replays that file. Nothing is compiled and nothing executes;
the trace is a description of work and the simulator prices it.

That still works and is unchanged. What is new is that **the same mapped IR now
continues down to a program**. Two new dialect conversions turn mapped
bit-serial operations into concrete DRAM commands and then into address
arithmetic and inline machine instructions; the result goes through the ordinary
LLVM pipeline into a static x86 binary, and that binary runs under gem5's
syscall-emulation mode, issuing each row operation as an instruction. The memory
sees the same commands in the same order, but they now arrive from a processor
that had to compute their addresses and get them into the store queue — a cost
the trace replay cannot express at all.

---

## 2. A worked example, from source to simulation

Everything below is a real command with its real output. Run them in order and
you will end up with the same binary and the same numbers. The kernel is
`vadd` — elementwise `a + b` over a million `i16` values — mapped onto 32 banks
of DDR4, because it is the smallest thing that exercises every stage while
staying readable.

The one-command version of this whole section is

```sh
MIMDRAM/microworkloads/pud/compile_pud_binary.sh \
    Cinnamon/limon_benchmark/linalg_vadd_1048576.mlir -n 32 -m ddr
```

Do that once you have read what it does. Adding `--keep` makes it print the
directory it left every intermediate in, so you can follow along against your
own run.

For the commands below, set:

```sh
CINM=Cinnamon/cinnamon/build-pud/bin/cinm-opt
LLVM=Cinnamon/llvm/build/bin
K=Cinnamon/limon_benchmark/linalg_vadd_1048576.mlir
W=$(mktemp -d)          # somewhere to put the intermediates
```

### The input

```mlir
func.func @vadd(%a: tensor<1048576xi16>, %b: tensor<1048576xi16>,
                %c: tensor<1048576xi16>) -> tensor<1048576xi16> {
  %0 = linalg.add ins(%a, %b : tensor<1048576xi16>, tensor<1048576xi16>)
                  outs(%c : tensor<1048576xi16>) -> tensor<1048576xi16>
  return %0 : tensor<1048576xi16>
}
```

Ordinary `linalg` on ranked tensors. No annotations, nothing PuD-specific — the
whole point is that this is the input a compiler gets.

---

### Step 1 — `--convert-linalg-to-bits`

```sh
$CINM $K --convert-linalg-to-bits -o $W/s1.mlir
```

```mlir
module {
  func.func @vadd(%arg0: tensor<1048576xi16>, ...) -> tensor<1048576xi16> {
    %0 = tensor.empty() : tensor<128x!bits.slice<16x8192>>
    %1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %0) -> (...) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3] [8192] [128] : ...
      %3 = bits.transpose %extracted_slice : tensor<8192xi16> -> !bits.slice<16x8192>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg3] [8192] [128] : ...
      %4 = bits.transpose %extracted_slice_0 : tensor<8192xi16> -> !bits.slice<16x8192>
      %5 = bits.add_i %3, %4 : !bits.slice<16x8192>, ... -> !bits.slice<16x8192>
      %inserted = tensor.insert %5 into %arg4[%arg3] : tensor<128x!bits.slice<16x8192>>
      scf.yield %inserted
    }
    %2 = bits.assemble %1 : tensor<128x!bits.slice<16x8192>> -> tensor<8192x128xi16>
    %collapsed = tensor.collapse_shape %2 [[0, 1]] : ... into tensor<1048576xi16>
    return %collapsed
  }
}
```

**What happened.** The million-element vector was cut into **128 tiles of 8192
elements**, and each tile became a `!bits.slice<16x8192>`.

That type is the centre of everything downstream. It is 8192 values of 16 bits,
**stored transposed**: 16 DRAM rows, row *i* holding bit *i* of all 8192 values.
A DRAM row operation works on a whole row at once, so this layout is what turns
"add 8192 numbers" into "16 row operations' worth of work" instead of 8192
separate additions. 8192 is not arbitrary — it is a DDR4 row in bits, one bit
per column.

`bits.transpose` and `bits.assemble` mark where the layout changes. They are not
lowered to anything later on: a compiled kernel computes on rows that are
*already* bit-sliced, and getting ordinary data into and out of that layout is
the missing piece described in section 7.

`bits.add_i` is still one operation on a whole slice. The frontend has chosen
the tiling and the layout; it has not yet said anything about banks or commands.

Some kernels take frontend options here — a convolution takes
`conv-lane-dim={oc|spatial}` to choose which axis fills a row. `vadd` has only
one sensible choice, so there is nothing to pass.

---

### Step 2 — `--bits-opt-mapping`

```sh
$CINM $K --convert-linalg-to-bits \
    "--bits-opt-mapping=num-banks=32 mem-type=ddr emit-trace=false" \
    -o $W/s2.mlir
```

This is the ILP, and it is the part the paper is about. It prints a page of
Gurobi solver output to stderr, ending with the cost of the schedule it chose:

```
Optimize a model with 3 rows, 35 columns and 36 nonzeros (Min)
...
Optimal solution found (tolerance 1.00e-04)
Best objective 5.160000000000e+02, best bound 5.160000000000e+02, gap 0.0000%
```

Append `2>/dev/null` if you do not want it. **If instead you see `Gurobi error:
License ... has expired` followed by `no bank mapping on any enclosing loop`,
that is the licence and not a compiler bug** — the mapping is upstream of
everything, so nothing downstream can run without it.

**The options:**

| option | meaning |
|---|---|
| `num-banks=32` | how many banks the kernel may spread over. This is the *kernel's* bank budget, not the machine's size. |
| `mem-type=ddr` | `ddr` or `hbm`. Sets banks per channel (32 or 16) and the row stride (8192 or 1024 bytes). **It is baked into every address the kernel later computes**, so the simulation has to be run on the same memory. |
| `emit-trace=false` | **the only difference from how the paper invokes this pass.** By default it also writes the trace — the row-op sequence the schedule would issue — to a fixed path. The compiled path wants the decisions, not the file. (That fixed path is also why two builds cannot run through this stage at the same time.) |

**What happened.** The IR is structurally unchanged; the pass writes its
decisions on as attributes:

```mlir
module attributes {bits.memory_geometry = {banks_per_channel = 32 : i64,
    ranks_per_channel = 1 : i64, row_stride = 8192 : i64, total_banks = 32 : i64}} {
  ...
    scf.yield %inserted
  } {bits.bank_group_stride = 32 : i64, bits.bank_id = 0 : i64,
     bits.concurrent_groups = 1 : i64, bits.iters_per_bank = 4 : i64,
     bits.loop_kind = "parallel", bits.parallel_factor = 32 : i64,
     bits.wave_repeat = 1 : i64}
```

Read it as: the 128-iteration loop is **parallel**, it is spread across
**32 banks** (`parallel_factor`), each bank therefore runs **4** of the 128
tiles (`iters_per_bank`), and there is **one** such group (`concurrent_groups`,
`wave_repeat`). 32 x 4 = 128, which is the whole loop.

For a kernel with a reduction this is where the interesting decision is made —
how far to split the reduction across banks, trading copies between banks
against a shorter dependence chain. `vadd` has no reduction, so the mapping is
the obvious one and the ILP is trivially small (3 rows, 35 columns). Try
`num-banks=128` and watch `parallel_factor` and `iters_per_bank` move.

---

### Step 3 — `--bits-pattern-apply`

```sh
$CINM $K --convert-linalg-to-bits \
    "--bits-opt-mapping=num-banks=32 mem-type=ddr emit-trace=false" \
    --bits-pattern-apply -o $W/s3.mlir 2>/dev/null
```

```mlir
%5 = bits.create_all_zero_row %c8192_i64 : i64 -> !bits.row<8192>
%6 = bits.create_slice %c16_i64, %c8192_i64 : i64, i64 -> !bits.slice<16x8192>
%7:2 = scf.for %arg5 = %c0 to %c16 step %c1
       iter_args(%arg6 = %6, %arg7 = %5) -> (!bits.slice<16x8192>, !bits.row<8192>) {
  %9  = bits.extract_row %3, %8 : !bits.slice<16x8192>, i64 -> !bits.row<8192>
  %10 = bits.extract_row %4, %8 : ...
  %sum, %cout = bits.row_add %9, %10, %arg7 {bits.carry = "resident"}
       : !bits.row<8192>, !bits.row<8192>, !bits.row<8192> -> !bits.row<8192>, !bits.row<8192>
  %11 = bits.insert_row %arg6, %sum, %8 : ...
}
```

**What happened.** The one slice-level operation became **a loop over the 16 bit
positions**, carrying a carry row from one bit to the next — a ripple-carry
adder written as a loop. Every value is now exactly one DRAM row
(`!bits.row<8192>`).

This is where each kernel's arithmetic gets its shape: multiplication expands to
shift-and-add, subtraction to an adder with one input complemented, a reduction
to a tree or a chain, min/max to compare-then-select, the range scan to its own
sequence. `bits.carry = "resident"` records that the carry stays in a row of its
own rather than being rematerialised.

No commands yet — `bits.row_add` is still "add these two rows", not a sequence
of DRAM activations.

---

### Step 4 — `--convert-bits-rows-to-pud`

```sh
$CINM $K --convert-linalg-to-bits \
    "--bits-opt-mapping=num-banks=32 mem-type=ddr emit-trace=false" \
    --bits-pattern-apply \
    "--convert-bits-rows-to-pud=operand-per-wave=1" -o $W/s4.mlir 2>/dev/null
```

**The option.** `operand-per-wave=1` (the default) gives each wave its own rows
for the operand it reads, so the kernel computes the right answer.
`operand-per-wave=0` — `--shared-pass-rows` on the driver script — gives one set
of rows to all the waves of a pass, so they recompute on the same tile. That is
a faithful model of the schedule's *cost* and not a program: it exists because
a kernel whose operand does not fit in a subarray cannot be built any other way,
and building it as a cost model is better than not building it. The script tries
`1` first and falls back.

**What happened, part one: the loop nest now says the mapping out loud.**

```mlir
scf.for %arg0 = 0 to 1  step 1 {                 // concurrent_groups
  %1 = %arg0 * 4
  scf.for %arg1 = 0 to 4  step 1 {               // iters_per_bank
    %3 = %1 + %arg1                              //   which tile this is
    scf.for %arg2 = 0 to 32 step 1 {             // the banks, in parallel
      %5 = pud.get_ctrl_row DCC1, 0, 0, %4, 0 : ... -> !pud.row<B>
      %6 = pud.get_ctrl_row C_0,  0, 0, %4, 0 : ... -> !pud.row<C>
      pud.AAP %6, %5 {backend = #pud<backend simdram>} : !pud.row<C>, !pud.row<B>
    }
    scf.for %arg2 = 0 to 16 step 1 {             // the bit positions
      scf.for %arg3 = 0 to 32 step 1 {           //   again over the banks
        ...
```

Each loop is one of the mapping's numbers. The bank loop is innermost because
the banks work in parallel: a wave is "the same command to every bank", and
issuing it is a loop over banks. The first bank loop is the carry seed — every
bank's carry row is set to zero before the bit loop starts.

**What happened, part two: each row operation became commands.**

`pud.AAP` is activate-activate-precharge: copy a row into a group of rows and
let the group's majority settle. `pud.AP` is the single-row form.
`pud.get_ctrl_row` names one of the **18 rows the substrate reserves** — four
compute rows `T0..T3`, two dual-contact cells and their complements, and the row
groups that form a majority gate. `pud.get_row` names an ordinary data row. A
coordinate is `(channel, rank, bank, subarray, row)`.

Where do the sequences come from? **lime**, a logic synthesiser that solves for
the shortest command sequence on the substrate's model. For a few primitives
lime's scheduler is not shortest — xnor, xor, subtract, the range scan — and
those have hand-written sequences instead. Because a hand-written sequence has
no proof behind it, the compiler contains a **substrate simulator** that runs
every sequence over all input assignments *and all 64 initial states* of the
compute rows and cells, and refuses to compile if one does not compute what it
claims. It agrees with all thirteen of lime's machine-proved sequences, which is
what licenses it to vouch for the rest. There is a `rowop-sanity` tool that
prints every primitive's sequence beside its reference count.

**What happened, part three: the shape of the row space is published.**

```mlir
module attributes {..., pud.banks_per_channel = 32 : i64,
  pud.channel_size = 134217728 : i64, pud.data_rows_per_bank = 192 : i64,
  pud.row_stride = 8192 : i64, pud.total_banks = 32 : i64}
```

192 data rows per bank is what *this* kernel needs. The ceiling is **494**: every
command names a control row, the hardware requires a command's rows to share a
subarray, and the control rows sit at the start — so all of a kernel's data has
to live in subarray 0, which is 512 rows less the 18 reserved. When a kernel
does not fit, this pass says so by name instead of failing later.

---

### Step 5 — `--convert-pud-rows-to-func`, and into LLVM dialect

```sh
$CINM $W/s4.mlir \
    --convert-pud-rows-to-func \
    --convert-scf-to-cf --convert-arith-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts -o $W/s5.mlir
```

Only the first pass is ours. The other four are stock MLIR, and they are the
standard way out of the dialect tower:

| pass | what it does here |
|---|---|
| `--convert-pud-rows-to-func` | **ours.** Row coordinates become address arithmetic on the runtime's arena; `pud.AAP`/`pud.AP` become inline machine instructions. This is the last pass that knows anything about DRAM. |
| `--convert-scf-to-cf` | structured control flow to branches: the `scf.for` nests over waves, banks and bit positions become blocks and conditional branches. |
| `--convert-arith-to-llvm` | the `arith` dialect — every index computation in those loops — into the LLVM dialect. |
| `--convert-func-to-llvm` | `func.func` and `func.return` into `llvm.func` and `llvm.return`, and the calling convention with them. |
| `--reconcile-unrealized-casts` | removes the `unrealized_conversion_cast` placeholders the previous three leave at their boundaries. Without it the module still has casts between the old and new types and will not translate. |

The order matters: each of the middle three converts one dialect, and the
reconciliation has to come last, once nothing is left that needs a placeholder.
Note this invocation needs no `2>/dev/null` — it starts from `$W/s4.mlir`, which
is already mapped, so the ILP does not run again and nothing is printed.

**What happened: coordinates became addresses, commands became instructions.**

```mlir
llvm.mlir.global external @pud_arena() {addr_space = 0 : i32, dso_local} : !llvm.ptr
...
%33 = llvm.getelementptr inbounds %1[%32]  : (!llvm.ptr, i64) -> !llvm.ptr, i8
%37 = llvm.getelementptr inbounds %33[%36] : (!llvm.ptr, i64) -> !llvm.ptr, i8
...
llvm.inline_asm has_side_effects ".byte 0x0F, 0x04\0A.word 97",
    "{rdi},{rsi},~{memory}" %37, %47 : (!llvm.ptr, !llvm.ptr) -> ()
```

A row's address is arithmetic on one global, `pud_arena`, which the runtime
defines:

```
  address = arena
          + channel * channel_size                    <- its own getelementptr
          + row     * (row_stride * banks_per_channel)
          + bank    *  row_stride
```

Two deliberate choices, both measured rather than assumed. **Arithmetic, not a
lookup table**: publishing a table of row addresses costs a load per operand out
of a structure the size of the kernel's whole row space, which does not stay in
the first-level data cache, and every miss becomes a DRAM read in the middle of
the row-op stream. **The channel term gets its own `getelementptr`**: folded
into a single offset the sum also carries the row and the bank, so it cannot be
hoisted out of a loop over banks even though the channel does not change in
there — the difference is 6 instructions per row operation against 10.

The command is emitted **inline, not called**. `.word 97` is `0x61`, which is
AAP. A call would add a push and a pop of the return address to every row
operation, and the push is an ordinary store that orders against every other
store the program makes. `has_side_effects` and the `~{memory}` clobber are what
make it unmovable: what the command leaves behind is in DRAM rows, which nothing
in the IR can name, so without them LLVM would be free to delete or reorder the
whole schedule.

---

### Step 6 — LLVM

```sh
$LLVM/mlir-translate --mlir-to-llvmir $W/s5.mlir -o $W/kernel.ll
$LLVM/opt -O3 -unroll-threshold=0 $W/kernel.ll -o $W/kernel.opt.bc
$LLVM/llc -filetype=obj -relocation-model=static $W/kernel.opt.bc -o $W/kernel.o
```

Three tools, and what each one is for:

| | |
|---|---|
| `mlir-translate --mlir-to-llvmir` | translates the LLVM **dialect** into actual LLVM IR. It is not an optimisation or a conversion — the module is already entirely in the LLVM dialect by now, and this just changes the representation from MLIR's to LLVM's. The output is `.ll`, which is worth reading once. |
| `opt` | LLVM's optimiser. Runs over the kernel and writes bitcode. |
| `llc` | LLVM's backend. Bitcode to a native object file. |

And the flags:

| flag | why |
|---|---|
| `-O3` | the address arithmetic is what needs optimising, and there is a lot of it — one address per row operation, built out of a multiply and two adds. `-O3` folds it into addressing modes and hoists what is loop-invariant, which is most of it. It cannot touch the row operations themselves: they are inline assembly with side effects and an unnamed memory clobber, so the optimiser must keep every one of them, in order. |
| `-unroll-threshold=0` | see below. |
| `-filetype=obj` | emit an object file rather than assembly. `-filetype=asm` is the readable version if you want to see the loop without `objdump`'s trouble decoding the row op. |
| `-relocation-model=static` | no position-independent code. The binary is statically linked and loaded at a fixed address under syscall emulation, and a global offset table would put one more load in the row-op path for nothing. |
| `--x86-experimental-pref-innermost-loop-alignment=6` | **not used here**, see below. |

**`-unroll-threshold=0`** — measured, not assumed. A loop body here is a handful
of instructions issuing a DRAM command, so the program is as long as the
schedule and is read once per run. Unrolling trades instruction fetch for issue,
and at this shape the trade is bad: fully unrolled, popcount takes 2.3x the time
and 20947 instruction-cache misses against 10.

**`-relocation-model=static`** — the binary runs under syscall emulation with no
dynamic loader.

**`--x86-experimental-pref-innermost-loop-alignment=6`** — *not used here*, and
that is deliberate. The driver script adds it **only when the bank count exceeds
one channel's worth** (more than 32 on DDR4, 16 on HBM). See section 4 for why
it is conditional; on this 32-bank example it would make things worse.

Look at what came out:

```sh
objdump -d $W/kernel.o | sed -n '/<vadd>:/,$p'
```

```
  d0:   0f 04 61 00                 <- the row op: 0F 04, then .word 0x0061 = AAP
  d4:   49 ff c1                    inc %r9
  d7:   48 81 c6 00 20 00 00        add $0x2000, %rsi     <- next bank, source
  de:   48 81 c7 00 20 00 00        add $0x2000, %rdi     <- next bank, destination
  e5:   49 83 f9 1f                 cmp $0x1f, %r9        <- 31: the 32 banks
  e9:   72 e5                       jb  d0
```

That is the innermost loop of the whole program: **six instructions per row
operation**, one of which is the row operation. `0x2000` is 8192, the DDR4 row
stride, so advancing one bank is one add. objdump does not know `0f 04` and will
print `(bad)` and then resynchronise in the middle of the next instruction, so
read the bytes rather than its mnemonics.

Those six instructions are the whole of the host-side cost discussed in
section 6.

---

### Step 7 — link

The link happens **inside the container**, because the binary has to start under
syscall emulation and the container's older toolchain stays within what that
supports. The container sees the MIMDRAM directory as `/workspace`, so the
object has to be somewhere under it first:

```sh
cp $W/kernel.o MIMDRAM/microworkloads/pud/build/kernel.o

docker exec mimdram bash -c "cd /workspace/microworkloads && \
  gcc -static -std=c99 -O2 -msse2 -I ../gem5/util/m5 -I . \
    -DPUD_KERNEL=vadd \
    -DPUD_DATA_ROWS=192 -DPUD_BANKS_PER_CHANNEL=32 -DPUD_TOTAL_BANKS=32 \
    -DPUD_CHANNEL_SIZE=134217728 -DPUD_ROW_SIZE=8192 \
    m5op_x86.S rowop.S pud/pud_main.c pud/pud_runtime.c pud/build/kernel.o \
    -o pud/build/vadd_ddr_32.exe"
```

**The compiler flags.** These are inherited almost verbatim from
`microworkloads/Makefile`, which is how the hand-written PuD microbenchmarks in
that directory have always been built; the row-op stubs and the statistics hooks
are the same ones.

| flag | why |
|---|---|
| `-static` | there is no dynamic loader under syscall emulation. Everything the program will ever call has to be in the file. |
| `-std=c99` | the container's compiler is gcc 4.8, whose default is `gnu89`. The runtime is C99 and asks for `posix_memalign` by defining `_POSIX_C_SOURCE` itself. |
| `-O2` | for the runtime and the driver only. **It cannot touch the kernel**: that arrived as `kernel.o`, already compiled by `llc` in step 6, and gcc only links it. |
| `-msse2` | carried over from the Makefile recipe. On x86-64 SSE2 is part of the baseline ABI, so it is a no-op here; it is kept so that the two build recipes stay comparable. |
| `-I ../gem5/util/m5` | where `m5op.h` is — the statistics hooks the driver calls. |
| `-I .` | the compile runs from `microworkloads/`, and this is where `mimdram.h` is, which the runtime includes. |
| `-o pud/build/...` | the output **must land under `/workspace`**, because that is the only thing the container — and therefore gem5 — can see. |

**The defines.** Every one of them is a number read straight out of the module
attributes the compiler published in step 4. That is the contract between the
two sides: the compiler decided the shape of the row space, and the runtime lays
out exactly that shape. Each is checked — the runtime `#error`s if any is
missing, refuses a data-row count below one, and refuses a stride that is not a
power of two of at least a burst.

| define | from | what it does |
|---|---|---|
| `PUD_KERNEL=vadd` | the `func.func` symbol in the generated IR | the driver declares `extern void PUD_KERNEL(void)` and calls it. This is the only thing tying the driver to this particular kernel; the script reads the symbol out of the IR, and `-k` overrides it. |
| `PUD_DATA_ROWS=192` | `pud.data_rows_per_bank` | how many data rows per bank the kernel addresses, on top of the 18 reserved. The arena is sized from it. Too small and the kernel walks off the end of its own memory — which is a bug that was found exactly this way. |
| `PUD_BANKS_PER_CHANNEL=32` | `pud.banks_per_channel` | the **channel's** bank count, 32 on DDR4 and 16 on HBM — not the kernel's budget. It sets the period at which consecutive rows repeat, because that is the period the memory controller decodes them at. A kernel using 8 banks of a 32-bank channel still has to leave the other 24 a row's worth of space each, or its second row lands in bank 8. |
| `PUD_TOTAL_BANKS=32` | `pud.total_banks` | how many banks this kernel spreads over, which is what the mapping was given. Divided by the previous number, it says how many channels the arena has to reach across. |
| `PUD_CHANNEL_SIZE=134217728` | `pud.channel_size` | 128 MB, one DDR4 channel. A bank of channel *c* sits *c* whole channels further into the address space, because that is where the controller owning it begins, so the arena has to be this much wider per extra channel. |
| `PUD_ROW_SIZE=8192` | `pud.row_stride` | the row buffer of the memory this kernel was mapped for — 8 kB on DDR4 x64, 1 kB on HBM. The kernel already baked this into every address it computes, so the runtime lays the rows out at whatever it is told rather than at a constant of its own. It used to be checked against `mimdram.h`'s fixed notion of a row, which is DDR4's, and that check made HBM kernels impossible to build. |

**What each source file contributes:**

- `pud/build/kernel.o` — the kernel from step 6. Nothing but row operations and
  the arithmetic to address them.
- `pud/pud_runtime.c` — allocates the arena in **one** aligned block, with the 18
  control rows first, in the order the compiler enumerates them, and the data
  rows after; publishes `pud_arena`. One block rather than a row at a time: an
  aligned allocation of exactly one period costs the allocator more than a
  period, so per-row allocation spread the rows over about twice the address
  space they need, and a kernel with a few hundred rows would spill past the end
  of its channel — where the wrong controller decodes it.
- `pud/pud_main.c` — calls `pud_init`, runs the kernel once **cold**, then
  `m5_reset_stats`, runs it again, `m5_dump_stats`. The cold run matters: the
  program is as long as its schedule, and reading it in for the first time is a
  demand on the same memory the row operations are going to, so without it the
  cost of fetching the program is counted as the cost of issuing the schedule.
  Running it twice is sound only because row operations have no functional
  effect in this model — see section 7.
- `rowop.S` — the row-op opcodes: `0x0F 0x04` plus a two-byte selector, one
  symbol per operation. The compiler does not call these; it emits the same two
  bytes inline. They are here because the legacy microbenchmarks call them, and
  because having the symbols makes the encoding one grep away.
- `m5op_x86.S` — gem5's statistics hooks, `m5_reset_stats` and `m5_dump_stats`.

---

### Step 8 — simulate

```sh
docker exec mimdram bash -c "cd /workspace && \
  gem5/build/X86_PUD/gem5.opt --outdir=/tmp/out \
    gem5/configs/dram/cpu_rowop.py \
    --cmd=/workspace/microworkloads/pud/build/vadd_ddr_32.exe \
    --mem-type=DDR4_2400_x64 --cpu-type=o3"
```

| option | meaning |
|---|---|
| `--cmd` | the static binary to run. Must be visible inside the container, i.e. under `/workspace`. |
| `--mem-type` | `DDR4_2400_x64` or `HBM3_4Gb_x64`. **Must agree with the `mem-type` the kernel was mapped for** — the row stride is baked into its addresses. See the note below. |
| `--cpu-type` | `o3` (out-of-order, what every measurement uses) or `timing`. |
| `--cpu-clock` | default `2GHz`. Worth sweeping: it is how you tell a memory-bound kernel from a host-bound one. |
| `--machine-banks` | **the size of the simulated machine**, default 128, *not* the kernel's bank budget. A 32-bank kernel on the default machine occupies banks 0-31 of channel 0, which is where the trace player puts them too. Do not set this to your kernel's bank count. |
| `--sq-entries` | store-queue depth, 0 = leave the out-of-order default. |
| `--l1i-size`, `--l1d-size`, `--no-caches` | diagnostics. Widening the instruction cache is how the instruction-fetch effect in section 6 was isolated. |

**If you cross the memory types, you will not be told so in as many words.**
The binary does not record what it was built for and gem5 cannot ask, so the
only thing that catches a mismatch is the addresses failing to decode — which
they do, quickly and loudly, but as an internal complaint rather than an
explanation. An HBM kernel run on DDR4 aborts on
`Assertion 'dram_pkt->rank == dram_pkt1->rank' failed` in `dram_ctrl.cc`; a DDR4
kernel run on HBM aborts a few lines earlier, on a row address that is outside
the memory the controller owns. Both happen within a few tens of microseconds of
simulated time, so you find out in seconds rather than after a long run. If you
see either, check the `-m` the kernel was compiled with before you look for a
bug in the memory model. The default name the build script gives a binary —
`pud_<kernel>_<mem>_<banks>.exe` — carries the answer.

The same build also runs the trace player —
`gem5/configs/dram/trace_player.py --trace=... --mem-type=... --backend=simdram`
— so both paths can be measured on one simulator, which is what makes the
comparison in section 6 apples to apples.

---

### Step 9 — read the result

```sh
docker exec mimdram awk '/Begin Simulation Statistics/{b++} b==1{
       if($1~/rowOpCount$/)                      ops += $2
       if($1~/rowOpFirstTick$/ && $2+0>0 && (f==""||$2+0<f)) f = $2+0
       if($1~/rowOpLastTick$/  && $2+0>l)        l = $2+0 }
     END{ printf "%d row ops, %.3f us\n", ops, (l-f)/1e6 }' /tmp/out/stats.txt
```

```
16512 row ops, 28.887 us
```

That is with the runtime that registers its arena (MIMDRAM `4385f03`, section 7).
Before it, the same kernel measured 28.711 us. The schedule is the same 16512
operations and they end within 18 ns of each other; what moved is the start of
the window. The runtime grew by one call, the kernel linked after it moved 32
bytes, and whether one load misses the cache just before the first row op
changed with it. The same binary gives the same number with the effect on or
off, so this is layout, not the memory model -- the same sensitivity the
alignment flag in section 4 is about.

**Expect the makespan to wobble by a few parts in ten thousand, and do not
chase it.** The row-op count is exact and reproducible; the makespan is
reproducible only for a fixed `--cmd` *string*. Under syscall emulation gem5
lays argv on the process stack, so the length of the path you pass moves the
initial stack pointer, which moves the heap, which moves where the runtime's
aligned arena physically lands — and that changes which banks and rows the
kernel's addresses decode to. Measured on this very binary: run it as
`.../build/vadd_ddr_32.exe` and it takes 28.711 us; run the byte-identical copy
as `.../build/verify_0922/vadd_ddr_32.exe`, twelve characters longer, and it
takes 28.721. Two different paths of the *same* length give the same number to
the tick. 0.03% either way, and the row-op count never moves.

**Two traps, both of which silently give a wrong number.** The file holds **two**
dumps — one at `m5_dump_stats` and one at exit — so only the first block is the
measured run; that is what `b==1` is for. And `rowOpCount` is per memory
controller and has to be summed, while `rowOpFirstTick` reads zero on a
controller that saw no row operation, so the minimum has to skip the zeros.

**Check the count by hand.** An *n*-bit addition on this substrate costs
`8n + 1` row operations, so a 16-bit add is 129. There are 128 tiles. 128 x 129
= **16512**, which is what came out, and is also exactly what the trace player
reports for the same kernel. The three numbers that have to agree — what the
mapping priced, what the binary issued, and what the trace expands to — agree
here and in all 140 configurations of section 6.

The makespan, last row-op tick minus first, is the number every comparison is
made on.

---

## 3. The contract between the two repositories

Four things, and nothing else:

**1. The instruction.** `MIMDRAM/microworkloads/rowop.S` defines the opcodes;
`0x0F 0x04` followed by a two-byte selector, with the rows in `rdi`, `rsi`,
`rdx`. `0x60` is AP, `0x61` is AAP; the rest cover the older primitives and the
two halves of a cross-channel copy. gem5 decodes it in
`arch/x86/isa/decoder/two_byte_opcodes.isa`, and
`cpu/base_dyn_inst.hh` turns one instruction into **three address translations**
and a single write-queue entry carrying `{op, dest, src1, src2}`.

**2. The arena.** `pud_arena`, a `char *` the runtime defines and the kernel
reads once per function. The runtime allocates every row in one aligned block so
that consecutive rows sit one *channel-bank period* apart — `row_stride *
banks_per_channel` — because that is the period the memory controller decodes
them at. A kernel given 8 banks of a 32-bank channel still leaves the other 24 a
row's worth of space each, or its second row lands in bank 8.

**3. The shape of the row space**, which the compiler publishes as module
attributes and the build script passes to the runtime as `-D` defines. A
mismatch is a build error rather than a wild pointer at run time.

**4. The order of the 18 control rows.** The compiler enumerates them; the
runtime lays them out first, in that order, with the data rows after. The
enumeration in `pud_runtime.c` must match the compiler's, and today it does.

**5. Where the rows start, told to gem5.** `pud_init` calls `m5_pud_arena` (m5
op `0x56`) with the arena, the period, the row stride and the channel size.
gem5 translates the address once (`sim/pseudo_inst.cc`) and keeps it in
`mem/pud_arena.{hh,cc}`, which also holds the table of which physical rows each
of the 18 names drives -- SIMDRAM's groups, as the compiler has them. That is
what lets a row op act on the rows and not only take the time. A program that
does not register gets no effect, exactly as before.

---

## 4. What changed in each repository

### MIMDRAM — 22 commits

- **A guest side that did not exist**: the runtime that owns the arena, the
  row-op stubs, an SE-mode configuration (`gem5/configs/dram/cpu_rowop.py`), and
  caches in front of the core so the program is fetched like a program.
- **Row addressing made to match the machine.** The row stride comes from the
  configured memory rather than being DDR4's; rows are spaced at the channel's
  bank count; the arena starts where a channel starts. Each was a real bug found
  by a kernel landing somewhere it did not intend.
- **Row operations routed to the controller that owns their rows**, which is
  what makes more than one channel possible at all.
- **Timing corrections**, each measured: the memory model no longer serialises
  bank activations it can overlap; a refresh that missed its slot catches up
  instead of aborting, and lands clear of its own power event; the statistics
  measure the span the row operations occupy rather than the whole run.
- **Instrumentation**: per-controller row-op counts and first/last ticks, a
  settable store-queue depth, clearer failures when an address is not a row.

Two of these are worth naming because they were subtle. `ROW_SIZE` was a
compile-time constant of 8192 used to round a row op's addresses before
translation; on DDR4 that *is* the row and the rounding does nothing, but HBM's
row is 1024 bytes, so it cleared three bits of the bank field and eight of the
kernel's banks arrived at one of the controller's. HBM was 3.7x slow for weeks
and looked like a timing problem. Separately, a controller owns one aligned
channel-sized window, so a kernel's rows land correctly only if the arena starts
near a channel boundary — which the guest cannot arrange, since it sees a
virtual address. Physical allocation is aligned from the configuration instead.

### Cinnamon — 132 commits

Stages 3 to 5 above are all new; that is most of it. The themes, in the order
they were fought: emitting row operations rather than calling them and resolving
addresses arithmetically; following the mapping's bank groups, waves and
reduction splits; making consecutive loop nests agree about where the data is;
living inside the 494-row subarray ceiling; and finally the host-side cost of
issuing a schedule.

**The last change is two commits, one in each repository, and neither works
alone.** Because a bank range never crosses a channel, the emitter used to write
one copy of every row-op loop per channel: one kernel's code grew from 34 kB at
32 banks to 138 kB at 128, against a 32 kB instruction cache, and every miss
became a DRAM read into the very controller running the row operations. Folding
those copies into a single loop with the channel as an induction variable
(Cinnamon `2ed389f`) leaves one instruction-fetch alignment where several used
to average out, so that one now has to be the good one — hence the alignment
flag in the build script (MIMDRAM `acbeaf7`). Applied to a kernel that stays
inside one channel, the same padding instead pushes it out of the instruction
cache and doubles its runtime, which is why it is conditional on the bank count
passing a channel. **Rebuild from the compiler commit with your own idea of how
to invoke `llc` and the numbers in section 6 do not come out.**

---

## 5. Getting it and building it

### What to pull

Both repositories have a `pud-codegen` branch; that is the whole handover.

```sh
# Cinnamon
git fetch origin
git checkout pud-codegen
git submodule sync --recursive          # the lime remote moved on this branch
git submodule update --init --recursive

# MIMDRAM -- the branch is on CapZTr/MIMDRAM, not on CMU-SAFARI's upstream
git clone -b pud-codegen https://github.com/CapZTr/MIMDRAM.git
```

**`git submodule sync` is not optional on Cinnamon.** This branch re-points
`cinnamon/extern/lime` from `CapZTr/lime` to `joaopaulocl/lime` and pins a
different commit. Without the sync you silently keep the old remote, the update
fetches a commit that is not there, and the failure does not name the cause.
The two submodules are:

| submodule | remote | pinned |
|---|---|---|
| `cinnamon/extern/lime` | `joaopaulocl/lime` | `1fceb9d` |
| `cinnamon/extern/eggmock` | `jungnitz/eggmock` | unchanged |

The kernels come with Cinnamon, in `limon_benchmark/` — the fourteen this path
is built and measured on, with a README saying what each one is. Nothing else
is needed from outside the two repositories.

### Building Cinnamon

Prerequisites beyond the usual: **LLVM/MLIR 19.1.3** with
`patches/llvm/linalg_reduce_fix.patch` applied, built somewhere you can point at
— the code includes `llvm/Support/LogicalResult.h`, which LLVM 18 does not have,
so the justfile's `llvmorg-18.1.6` comment is stale; a **Rust toolchain**,
because both submodules are Rust crates imported through Corrosion; the
**COIN-OR CBC** libraries (`coinor-libcbc-dev` on Ubuntu), which lime links
against; and **network access during configure**, because Corrosion itself is
fetched at configure time. Also **Gurobi 13.0** at `/opt/gurobi1300/linux64`,
where `lib/Dialect/Bits/Transforms/CMakeLists.txt` links it from — a build-time
dependency — and a **licence** that the C++ API accepts, at run time. The
size-limited licence that ships with pip `gurobipy` is refused
(`PIP license can only be used from gurobipy interface`), and inside a
container only a WLS licence works; a named-user one is locked to its host.
`cinm-opt` exits 0 even when Gurobi fails, so the first thing that notices is
`compile_pud_binary.sh` reporting that the IR is missing the shape of the row
space.

### Or: in Docker

`docker/` holds two images that reproduce all of this, verified on `vadd` to
the same 16512 row ops as section 2:

| image | from | holds |
|---|---|---|
| `cinnamon-pud` | Ubuntu 24.04 | clang, lld, Rust, CBC, Gurobi 13.0 — tools only |
| `mimdram-gem5` | Ubuntu 14.04 | gcc 4.8, Python 2, scons: gem5 and the final link |

LLVM, `cinm-opt` and gem5 are built into the bind-mounted trees rather than
into the images, so they live next to the sources as in a native build.

```sh
# once: images, the container compile_pud_binary.sh links in, gem5
docker build -t cinnamon-pud -f docker/cinnamon.Dockerfile docker
docker build -t mimdram-gem5 -f docker/mimdram.Dockerfile docker
docker run -d --name mimdram --user $(id -u):$(id -g) \
    -v $PWD/../MIMDRAM:/workspace mimdram-gem5 sleep infinity
docker exec mimdram bash -c 'cd gem5 && scons build/X86_PUD/gem5.opt -j8'

# LLVM, into llvm/build where compile_pud_binary.sh expects it
git clone --depth 1 -b llvmorg-19.1.3 https://github.com/llvm/llvm-project llvm
git -C llvm apply ../patches/llvm/linalg_reduce_fix.patch
docker/run.sh cmake -S llvm/llvm -B llvm/build -G Ninja \
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DBUILD_SHARED_LIBS=ON -DLLVM_USE_LINKER=lld -DLLVM_OPTIMIZED_TABLEGEN=ON
docker/run.sh ninja -C llvm/build

# every other command on the Cinnamon side
export GRB_LICENSE_FILE=/path/to/gurobi.lic     # a WLS licence
docker/run.sh <command>
```

`docker/run.sh` runs as you, mounts the workspace at its host path, and passes
the Docker socket through so that `compile_pud_binary.sh` can reach `mimdram`.
The commands below are the same, prefixed with it.

Configure once, into a directory of its own:

```sh
cmake -S cinnamon -B cinnamon/build-pud -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DLLVM_DIR=<llvm-build>/lib/cmake/llvm \
  -DMLIR_DIR=<llvm-build>/lib/cmake/mlir \
  -DCINM_BUILD_GPU_SUPPORT=OFF -DUPMEM_DIR=OFF -DTORCH_MLIR_DIR=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build cinnamon/build-pud --target cinm-opt
```

The first build is long; the Rust crates dominate it.

**After changing anything, rebuild incrementally — never reconfigure:**

```sh
ninja -C cinnamon/build-pud cinm-opt
```

That is the loop you will live in. A change to one of the conversion passes
relinks `cinm-opt` in well under a minute; the submodules are not rebuilt unless
you touch them. The repository's `justfile` has `just build` and `just cinm-opt`,
which do the same thing but against `cinnamon/build` — fine on a fresh clone,
but keep in mind that `just cleanBuild` deletes the build directory outright.

The passes live in `cinnamon/lib/Conversion/BitsToPuD/` (stage 4) and
`cinnamon/lib/Conversion/PuDToFunc/` (stage 5); the command sequences and the
substrate simulator are in `cinnamon/lib/Dialect/Bits/Codegen/RowOpLibrary.cpp`.
There is a `rowop-sanity` tool that prints every primitive's sequence beside its
reference command count, which is the fastest way to see whether a change to the
library did what you meant.

### Building MIMDRAM

You already have a working build of the paper version, so this is an incremental
rebuild, not a fresh one. The branch changes 22 files:

```
 gem5/build_opts/X86_PUD                             +3     a build directory of its own
 gem5/configs/dram/cpu_rowop.py                    +247     the SE-mode configuration (new)
 gem5/src/arch/x86/isa/decoder/two_byte_opcodes.isa  +7     decode AP and AAP
 gem5/src/arch/x86/isa/microops/rowbit.isa           ~6
 gem5/src/cpu/BaseCPU.py, base.cc, base.hh        +10/+7    the configured row size
 gem5/src/cpu/base_dyn_inst.hh                      ~17     one instruction -> three translations
 gem5/src/cpu/o3/lsq_unit.hh, lsq_unit_impl.hh      ~28     row-op overlap width, queue depth
 gem5/src/cpu/translation.hh                        ~12     route a row op to the owning controller
 gem5/src/mem/dram_ctrl.cc, .hh                    ~103     timing, refresh, per-controller stats
 gem5/src/mem/rowop_trace_player.cc                ~117     the range scan, and stats alongside
 gem5/src/sim/System.py, system.cc, system.hh       ~36     channel-aligned physical allocation
 microworkloads/pud/{compile_pud_binary.sh,pud_main.c,pud_runtime.c}  +346   the guest side (new)
 microworkloads/rowop.S                              +3     the two cross-channel opcodes
```

`build_opts/X86_PUD` is **byte-identical to `X86`** — same ISA, same CPU models,
same protocol. It exists only so that this work builds into its own directory
and leaves your paper build alone. So:

```sh
cd gem5
scons build/X86_PUD/gem5.opt -j8      # or build/X86/... to rebuild in place
```

scons rebuilds only what the changed files reach. The ISA decoder change does
force the x86 decode-generation step and its dependents, so budget more than a
trivial edit would suggest; the memory and CPU changes on their own are quick.

The paper's own build is untouched by any of this, and the two can coexist:
`build/X86/gem5.opt` replays traces exactly as it did, and `build/X86_PUD/gem5.opt`
runs compiled kernels **and** the trace player, which is what lets both paths be
measured on one simulator.

### Compiling and running a kernel

```sh
export CIMDRAM_ROOT=/path/to/workspace   # where Cinnamon and MIMDRAM sit side by side
MIMDRAM/microworkloads/pud/compile_pud_binary.sh kernel.mlir -n 128 -m ddr
```

`-n` bank count, `-m` `ddr`|`hbm`, `-o` output, `-k` kernel symbol,
`--shared-pass-rows` to build a kernel that does not fit as a cost model anyway,
`--keep` to keep the intermediates so you can read the IR at each stage.

The script expects Cinnamon and MIMDRAM to be siblings in one directory; it
takes `cinm-opt` from `Cinnamon/cinnamon/build-pud/bin`, the LLVM tools from
`Cinnamon/llvm/build/bin`, and it does the final link **inside the container**,
because the binary has to start under syscall emulation and the container's
older toolchain stays inside what that supports. Set `CIMDRAM_ROOT` if your
layout differs, and `PUD_CONTAINER` if your container is not named `mimdram`.

Then:

```sh
docker exec mimdram bash -c "cd /workspace && \
  gem5/build/X86_PUD/gem5.opt --outdir=/tmp/out \
    gem5/configs/dram/cpu_rowop.py \
    --cmd=/workspace/microworkloads/pud/build/<kernel>.exe \
    --mem-type=DDR4_2400_x64 --cpu-type=o3"
```

`--mem-type` is `DDR4_2400_x64` or `HBM3_4Gb_x64` and has to agree with the `-m`
the kernel was compiled for: the row stride is baked into its addresses.

**Reading the statistics.** Two traps, both of which will silently give you a
wrong number. The file holds **two** dumps — the driver runs the kernel once
cold, resets the statistics, runs it again, and gem5 dumps again at exit — so
only the first block is the measured run. And `rowOpCount` is per memory
controller and has to be summed, while `rowOpFirstTick` reads zero on a
controller that saw no row operation, so the minimum has to skip the zeros:

```sh
awk '/Begin Simulation Statistics/{b++} b==1{
       if($1~/rowOpCount$/)                      ops += $2
       if($1~/rowOpFirstTick$/ && $2+0>0 && (f==""||$2+0<f)) f = $2+0
       if($1~/rowOpLastTick$/  && $2+0>l)        l = $2+0 }
     END{ printf "%d row ops, %.3f us\n", ops, (l-f)/1e6 }' /tmp/out/stats.txt
```

The makespan that comes out of that — last row-op tick minus first — is the
number every comparison in section 6 is made on.

## 6. What state the path is in

Measured over **140 configurations** — 14 kernels x {DDR4, HBM3} x {8, 16, 32,
64, 128 banks} — each built, traced, and simulated down both paths on the same
simulator build.

**Everything builds.** 140 of 140. 83 keep each wave's own operand; 57 are too
large at that bank count and share one set of rows between a pass's waves.

**The row-op counts agree exactly, in all 140.** The compiled binary's static
schedule, the count gem5 observes, and the packets the trace player expands are
the same number everywhere. This is the result that matters: what the cost model
prices and what executes are the same thing.

**The makespans agree within one channel and diverge past it.**

```
  compiled / replayed        within one channel      across channels
    (70 configurations each)        1.004                 1.370
```

The divergence is **not a memory-model error**; it is the host, in two parts,
both measured. A **dispatch floor** of about 2.9 cycles per row operation — six
committed instructions retiring at IPC 2 with nothing stalling; raise the clock
and the compiled path meets the player exactly, because the memory is simply
faster than a 2 GHz core can feed it once there is more than one channel of it.
And **instruction-cache pressure**, which was the compiler's own doing and is
largely fixed (section 4). The last of it was the reduction tree writing out its
copy list in full inside the row loop; past one channel it now loops over it
(section 7, item 2).

> Re-checked on 2026-09-22, both ends. Six configurations rebuilt from the
> committed tree come out **byte-identical** to the binaries the matrix was
> measured on, and those binaries reproduce their recorded row-op counts and
> makespans to the digit.

### The one thing this path does *not* do

**No compiled kernel has ever been checked for numerical correctness.** It is
verified as a cost model — the right commands, in the right order, at the right
times — and nothing more.

Half of what stood in the way is done. gem5 used to model a row operation's
*timing* and not its *effect*: `ROWAAP`, the one thing this path emits, was an
empty stub. It now does what the compiler's substrate model says, a row wide
(MIMDRAM `4385f03`), and `pud_selftest.c` checks that on the primitives the
compiler emits — every bit of and, or, not, maj3, xor, xnor and the full adder,
in two channels, on DDR4 and HBM3.

What is missing is a kernel's data: `pud_main.c` still loads no operands and
checks no results, because nothing yet knows where a tensor element ends up
(section 7, item 1). Until that exists, treat a compiled kernel as a schedule
that executes, not as a program whose answers anyone has looked at.

---

## 7. What to do next

**1. Make a compiled kernel compute the right answer.** Three steps, in order:

- **Correct the control-row table** — done differently from what was planned
  here, in `4385f03`. The effect does not read the trace player's table: it has
  its own, in `mem/pud_arena.cc`, written from the compiler's
  (`getSimdramGroups`), so the groups that decide an answer are the
  compiler's. The player's table and `mimdram.h` keep Ambit's, and with them
  the player's sequences and the legacy microbenchmarks keep the names they
  were written against. The warning at the end of this section is therefore
  moot. What follows is the plan as it was.

  *Correct the control-row table on the trace player's side first.* The
  compiler and the guest runtime already use SIMDRAM's row groups — the ones in
  Figure 6 of the final published SIMDRAM paper, which redefined them relative
  to the arXiv version. `gem5/src/mem/rowop_trace_player.hh` and
  `microworkloads/mimdram.h` still carry Ambit's, and three of the four triple-row
  groups differ. Today this is inert, because nothing computes and the groups
  only have to be counted; the moment a command has an effect, which group it
  names decides the answer. The count stays at 18 and the trace format is
  unaffected.
  Note that the trace player's own command sequences were written against
  Ambit's groups. Leave them: the player is a cost oracle and is not worth
  converting into a correctness oracle. Correctness should be validated on the
  compiled path only.
- **Implement the effect** — done for AP, AAP and in-channel copies, in
  `4385f03`; see section 3, item 5. The 8192-byte assumption is gone for those
  three. Not done: a cross-channel copy (`ROW_RD_STREAM`/`ROW_WR_STREAM`) still
  moves no data, so a reduction that folds across channels gets the wrong
  answer — its two halves reach two controllers with nothing ordering one
  after the other. The legacy primitives keep `ROW_SIZE`.

  *As planned:* The compiler already contains the substrate model
  this needs — four compute rows, two dual-contact cells, majority on activation
  — proved against thirteen machine-checked sequences. gem5 needs the same model
  a row wide instead of a bit wide: bitwise over 64-bit words, with
  `maj(a,b,c) = (a&b)|(a&c)|(b&c)`, which already exists there as the majority
  primitive. Two related gaps: the functional paths still assume an 8192-byte
  row, which is wrong on HBM, and a cross-channel copy is modelled for timing
  only and moves no data.
- **Get data in and out.** Operands are bit-sliced across rows, banks and
  channels, so comparing against a reference needs the map from a tensor element
  to its location. The compiler knows it; the host does not. The right shape is
  for the compiler to **emit** pack and unpack routines from the same code that
  computes addresses, rather than writing them by hand on the host where they
  would drift from the mapping's decisions. **This is the largest unknown**:
  whether the packing the mapping chose is recoverable outside the pass has not
  been established.

Then a correctness suite on deliberately small shapes, kept separate from the
140-configuration matrix — small shapes for correctness, the matrix for cost.

**2. Take the rest of the host cost out** — done in code (`104f182`), not yet
accepted on the whole matrix. The reduction tree's copies are split into runs
that step both ends together, and runs that repeat across channels become one
loop nest over channels and banks. Within one channel they stay written out:
looping them there made gemv at 32 banks 0.65% slower, the same trade the
alignment flag in section 4 runs into.

Where it stands against the acceptance test — row-op counts equal in all 140,
no single-channel configuration slower:

- 100 of the 140 binaries do not change at all; only the four kernels that
  reduce across banks (bnn_dot, conv1x1_relu, gemv, knn_l2) do.
- Of those, the 20 single-channel binaries are **byte-identical** to before,
  so they cannot regress.
- The 20 multi-channel ones change. Measured so far: gemv, DDR4, 128 banks,
  494751 row ops either way, 992.244 → 977.258 us (−1.5%). An earlier version
  that also looped within a channel kept the row-op count in every one of the
  nine multi-channel configurations it finished and moved their makespans
  between −2.1% and +0.07%; the remaining multi-channel ones have not been run
  on the final version.

What is left is simulating those 20. Run them a few at a time: sixteen gem5
processes at once took down a 27 GB machine.

**3. Widen the coverage.** Only a fourteen-kernel set has been through this
path. GEMM never has — and it is also the one kernel the trace player cannot
replay, needing around 165 GB, so whether the compiled path can run it is
genuinely open. And only SIMDRAM is targeted; PRADA would need its command
sequences brought through stage 4.

One warning. **Do not correct the control-row table on its own as a tidy-up.**
By itself it changes no measured number, and it leaves the simulator
half-converted — the table naming one set of groups while the command sequences
around it assume another. It is worth doing as the first step of making kernels
compute, and not before.
